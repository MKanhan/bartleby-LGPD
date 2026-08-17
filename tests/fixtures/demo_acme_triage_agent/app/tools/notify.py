"""Email notification tool — sends classification result to claimant."""

from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage


def notify_claimant_email(to: str, nome: str, classification: str) -> bool:
    msg = EmailMessage()
    msg["From"] = os.environ["SMTP_USER"]
    msg["To"] = to
    msg["Subject"] = "ACME Seguros — Atualização do seu sinistro"
    msg.set_content(
        f"Olá {nome},\n\nClassificação preliminar: {classification[:300]}\n\n"
        "Atenciosamente,\nACME Seguros"
    )
    try:
        with smtplib.SMTP(os.environ["SMTP_HOST"], 587) as s:
            s.starttls()
            s.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
            s.send_message(msg)
        return True
    except Exception:
        return False
