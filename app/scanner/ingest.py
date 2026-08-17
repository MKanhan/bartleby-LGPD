"""Safe ingest of source code (zip or git) into a work directory."""

from __future__ import annotations

import contextlib
import ipaddress
import os
import shutil
import socket
import stat
import subprocess
import zipfile
from pathlib import Path
from urllib.parse import urlsplit

MAX_ZIP_SIZE_BYTES = 200 * 1024 * 1024
MAX_TOTAL_UNCOMPRESSED_BYTES = 500 * 1024 * 1024
MAX_ZIP_FILE_COUNT = 20_000
GIT_TIMEOUT_SECONDS = 60


class UnsafeZipError(ValueError):
    pass


class UnsafeGitUrlError(ValueError):
    pass


class IngestError(RuntimeError):
    pass


def _is_safe_path(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def _blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _host_ip_literals(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Parse host as an IP literal in every form a C resolver accepts: canonical
    IPv4/IPv6 plus non-canonical IPv4 (integer, hex, octal, shorthand via
    inet_aton) — these defeat a naive ipaddress-only check. Empty list = DNS name.
    """
    out: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    with contextlib.suppress(ValueError):
        out.append(ipaddress.ip_address(host))
    with contextlib.suppress(OSError, ValueError):
        out.append(ipaddress.ip_address(socket.inet_aton(host)))
    return out


def _assert_resolves_public(host: str) -> None:
    """Resolve a DNS hostname and reject if any address is private/internal —
    catches metadata.google.internal and attacker DNS pointing inward. (git
    re-resolves at clone time, so an egress firewall is the definitive control.)
    """
    try:
        infos = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except OSError as e:
        raise UnsafeGitUrlError(f"git host does not resolve: {host}") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if _blocked_ip(ip):
            raise UnsafeGitUrlError("git host resolves to a private/internal address")


def validate_git_url(url: str, host_allowlist: frozenset[str] | None = None) -> str:
    """Validate a user-supplied git clone URL against SSRF/RCE vectors.

    Only https is allowed; combined with GIT_ALLOW_PROTOCOL=https at clone time
    this defeats the ext::/file:///git://ssh:// transports (incl. transitive).
    Private/loopback/link-local/reserved IP-literal hosts are rejected to block
    SSRF to cloud metadata and internal services. An optional host allowlist
    locks SaaS deployments to known forges.
    """
    url = (url or "").strip()
    if not url or len(url) > 2048 or any(c in url for c in "\x00\n\r\t "):
        raise UnsafeGitUrlError("git URL contains invalid characters")
    parts = urlsplit(url)
    if parts.scheme != "https":
        raise UnsafeGitUrlError("only https git URLs are allowed")
    if "@" in parts.netloc:
        raise UnsafeGitUrlError("credentials in git URL are not allowed")
    host = parts.hostname or ""
    if not host:
        raise UnsafeGitUrlError("git URL is missing a host")
    # Block private/internal IP-literal hosts in any encoding (canonical, integer,
    # hex, octal, shorthand) — closes the SSRF-to-metadata/loopback vector.
    for ip in _host_ip_literals(host):
        if _blocked_ip(ip):
            raise UnsafeGitUrlError("private or internal git hosts are not allowed")
    if host_allowlist and host.lower() not in host_allowlist:
        raise UnsafeGitUrlError(f"git host not allowed: {host}")
    return url


def ingest_zip(zip_path: Path, work_dir: Path) -> Path:
    """Extract zip into work_dir/<stem> and return the extracted root path.

    Extracts member-by-member with explicit guards (no blind extractall):
    rejects absolute/traversal paths, symlink members (zip-slip), and decompression
    bombs (uncompressed-size and file-count caps).
    """
    zip_path = Path(zip_path)
    work_dir = Path(work_dir)
    if zip_path.stat().st_size > MAX_ZIP_SIZE_BYTES:
        raise IngestError(f"zip exceeds {MAX_ZIP_SIZE_BYTES} bytes")

    out = work_dir / zip_path.stem
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    total = 0
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
        if len(infos) > MAX_ZIP_FILE_COUNT:
            raise UnsafeZipError(f"zip has too many entries: {len(infos)}")
        for info in infos:
            name = info.filename
            if name.startswith("/") or ".." in Path(name).parts:
                raise UnsafeZipError(f"unsafe path in zip: {name}")
            # external_attr's high 16 bits hold the unix mode; many zips
            # (writestr, Windows) omit the S_IFMT type bits entirely (type_bits == 0).
            type_bits = (info.external_attr >> 16) & 0o170000
            if type_bits == stat.S_IFLNK:
                raise UnsafeZipError(f"symlink not allowed in zip: {name}")
            if type_bits not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise UnsafeZipError(f"unsupported entry type in zip: {name}")
            total += info.file_size
            if total > MAX_TOTAL_UNCOMPRESSED_BYTES:
                raise UnsafeZipError("zip uncompressed size exceeds cap")
            if not _is_safe_path(out, out / name):
                raise UnsafeZipError(f"unsafe path in zip: {name}")
            zf.extract(info, out)
            if not _is_safe_path(out, out / name):  # re-check after extract
                raise UnsafeZipError(f"path escape after extract: {name}")

    return out


def ingest_git(
    url: str,
    work_dir: Path,
    token: str | None = None,
    host_allowlist: frozenset[str] | None = None,
) -> Path:
    """Shallow-clone a validated git repo into work_dir and return the clone path."""
    validate_git_url(url, host_allowlist)
    host = urlsplit(url.strip()).hostname or ""
    # DNS names are resolved here (in the worker, off the request path) so a host
    # that points at an internal address is rejected before clone.
    if not _host_ip_literals(host):
        _assert_resolves_public(host)
    work_dir = Path(work_dir)
    safe_name = url.rstrip("/").split("/")[-1].replace(".git", "").replace(" ", "_") or "repo"
    out = work_dir / safe_name
    if out.exists():
        shutil.rmtree(out)

    clone_url = url
    if token and url.startswith("https://"):
        clone_url = url.replace("https://", f"https://{token}@", 1)

    # Hard-block non-https transports at the git layer (defends even if validation
    # is bypassed) and never prompt for credentials or read system/global config.
    env = {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ALLOW_PROTOCOL": "https",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
    }
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--", clone_url, str(out)],
            check=True,
            capture_output=True,
            timeout=GIT_TIMEOUT_SECONDS,
            env=env,
        )
    except subprocess.TimeoutExpired as e:
        raise IngestError(f"git clone timed out after {GIT_TIMEOUT_SECONDS}s") from e
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or b"").decode(errors="replace")
        raise IngestError(f"git clone failed: {stderr.strip()}") from e
    return out
