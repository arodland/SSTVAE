"""The reception aggregation server.

Many stations hear one transmission; each uploads the latents and
weights it demodulated, and the server combines every station's copy
into the best picture the network as a whole could hear. See
docs/reception-aggregation.md.

Importing this package pulls in nothing but the standard library and
numpy. FastAPI is imported by `app` alone, so the database, the
matching rule and the combining are all usable -- and testable --
without a web framework installed.
"""

from .config import ServerConfig

__all__ = ["ServerConfig"]
