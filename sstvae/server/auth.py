"""Per-station API keys.

The server operator issues a key per station; the station sends it as a
bearer token. Keys are stored as a sha256 hash, so the database is not
a list of credentials, and the plaintext is shown once at issuance and
never again.

The station's identity comes from the *key*, never from the payload. A
payload naming a different station than its key is refused rather than
believed -- otherwise any key would let its holder upload in anyone
else's name, and a station's reputation for good receptions would mean
nothing.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

__all__ = ["new_key", "hash_key", "key_matches"]


def new_key() -> str:
    """A fresh key. 32 bytes of urandom, url-safe."""
    return secrets.token_urlsafe(32)


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def key_matches(key: str, key_hash: str) -> bool:
    return hmac.compare_digest(hash_key(key), key_hash)
