"""Check that every download URL in croissant.json is reachable."""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

CROISSANT = Path(__file__).parents[1] / "croissant.json"
TIMEOUT = 60


def file_objects(path=CROISSANT):
    """Return ``(name, contentUrl)`` of every file object."""
    distribution = json.loads(path.read_text())["distribution"]
    return [
        (item["name"], item["contentUrl"])
        for item in distribution
        if "contentUrl" in item
    ]


def check(file_object):
    """Return an error message, or ``None`` if the URL serves data."""
    name, url = file_object
    # Some hosts reject HEAD, so request the first byte only.
    request = Request(
        url, headers={"Range": "bytes=0-0", "User-Agent": "fnirsatlas-url-check"}
    )
    try:
        with urlopen(request, timeout=TIMEOUT) as response:
            response.read(1)
    except OSError as error:
        return f"{name}: {url}: {error}"
    return None


def main():
    objects = file_objects()
    with ThreadPoolExecutor(max_workers=8) as pool:
        errors = [error for error in pool.map(check, objects) if error]
    for error in errors:
        print(error)
    print(f"{len(objects) - len(errors)}/{len(objects)} URLs reachable")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
