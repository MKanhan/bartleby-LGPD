"""agent-lgpd — LGPD documentation generator for AI agents."""

import warnings

# The vendored google-genai SDK emits a Pydantic ArbitraryTypeWarning at import
# (its HttpOptions model annotates a field with the builtin `any`); all of our
# own models are clean. Silence only that specific message (set here so it runs
# before the SDK is first imported).
warnings.filterwarnings("ignore", message=r"<built-in function any> is not a Python type.*")

__version__ = "0.4.5"
