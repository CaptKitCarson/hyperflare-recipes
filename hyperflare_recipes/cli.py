# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Command-line entrypoint for official Hyperflare model recipes."""

from pathlib import Path

from ignition.cli import main as ignition_main

from .recipes import recipes


def main(argv=None) -> int:
    package_root = Path(__file__).resolve().parent
    return ignition_main(
        argv,
        recipes=recipes(),
        prog="ignition",
        source_roots=(("hyperflare_recipes", package_root),),
    )


if __name__ == "__main__":
    raise SystemExit(main())
