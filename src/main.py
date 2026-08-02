from __future__ import annotations

import sys
from pathlib import Path

from parser import State, parse


def main() -> None:
    input_str = Path(sys.argv[1]).read_text()

    state = State()
    parse(input_str, state)

    print(state.qbits_dict, state.bits_dict)


if __name__ == "__main__":
    main()