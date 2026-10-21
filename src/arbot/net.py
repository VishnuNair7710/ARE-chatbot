"""TLS trust. Corporate networks often re-sign HTTPS traffic with a company certificate that is
installed in the operating system's trust store but unknown to Python's bundled CA list, which
makes every API and model download fail with CERTIFICATE_VERIFY_FAILED.

`use_system_certs()` makes Python trust the OS store (via `truststore`). Certificate
verification stays ON; we never disable it."""
from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)
_done = False


def use_system_certs() -> bool:
    global _done
    if _done:
        return True
    # Hugging Face's Xet transfer protocol fails behind some proxies ("Byte range not sequential");
    # the classic download path works everywhere.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    try:
        import truststore
    except ImportError:
        log.info("truststore not installed; using Python's default CA bundle")
        return False
    truststore.inject_into_ssl()
    _done = True
    return True
