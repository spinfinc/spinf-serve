"""python -m spinf_serve.server: the server from the environment (SPINF_MODEL, SPINF_PORT, ...), the same as `spinf-serve serve`."""

import sys

from spinf_serve.cli import main

if __name__ == "__main__":
    sys.exit(main(["serve", *sys.argv[1:]]))
