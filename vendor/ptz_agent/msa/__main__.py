"""Allow ``python -m msa`` to drop into the chat CLI."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
