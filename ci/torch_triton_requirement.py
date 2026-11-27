"""Print the CUDA wheel's Triton requirement for the installed CPU torch release."""

import json
from importlib.metadata import version
from urllib.request import urlopen

from packaging.requirements import Requirement


def main():
    release = version("torch").split("+")[0]
    with urlopen(f"https://pypi.org/pypi/torch/{release}/json", timeout=30) as response:
        requirements = map(Requirement, json.load(response)["info"]["requires_dist"])
        (selected,) = [
            item
            for item in requirements
            if item.name == "triton" and (not item.marker or item.marker.evaluate())
        ]
    print(f"triton{selected.specifier}")


if __name__ == "__main__":
    main()
