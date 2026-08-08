from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .parser import State, parse


def main() -> None:
    parser = argparse.ArgumentParser(prog="quantum")
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to a QASM file. Defaults to test.qasm in the project root.",
    )
    args = parser.parse_args(sys.argv[1:])

    input_str = args.input_file.read_text()

    state = State()
    parse(input_str, state)


if __name__ == "__main__":
    main()
