"""URL policy isolated from the navigation module cached by older deployments."""

from collections.abc import Iterable
from urllib.parse import unquote, urlsplit


def is_experimental_url(url: str | None, experimental_paths: Iterable[str]) -> bool:
    """Recognize a registered experimental URL, including a deployment base path."""
    if not url:
        return False
    pathname = unquote(urlsplit(url).path).rstrip("/").rsplit("/", 1)[-1]
    return bool(pathname) and pathname in experimental_paths
