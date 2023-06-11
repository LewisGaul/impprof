#!/usr/bin/env python3

from __future__ import annotations

import argparse
import dataclasses
import inspect
import logging
import re
import runpy
import sys
import time
from dataclasses import dataclass
from types import ModuleType
from typing import Iterator
from unittest import mock


IGNORE_PACKAGES = []

LOG_FILE = f".impprof.{int(time.time())}.log"

logger = logging.getLogger("impprof")

import_actual = __import__


def _get_calling_module() -> tuple[str, int]:
    try:
        calling_frame = inspect.currentframe().f_back.f_back
        frame_info = inspect.getframeinfo(calling_frame)
        frame_module = calling_frame.f_globals["__name__"]
        if frame_module.startswith("importlib._bootstrap"):
            return "<unknown>", 0
        return frame_module, frame_info.lineno
    except Exception:
        return "<none>", 0


def _log_event(msg: str) -> None:
    logger.debug("%d -- %s", time.time_ns() // 1000, msg)


def import_mock(
    name: str,
    globals=None,
    locals=None,
    fromlist=(),
    level: int = 0,
) -> ModuleType:
    if any(name == p or name.startswith(f"{p}.") for p in IGNORE_PACKAGES):
        return import_actual(name, globals, locals, fromlist, level)

    cached = name in sys.modules

    parent_mod, parent_mod_lineno = _get_calling_module()
    if parent_mod.startswith("importlib._bootstrap"):
        return import_actual(name, globals, locals, fromlist, level)

    if level == 0:
        import_mod = name
        full_import_mod = name
    else:
        # Add the parent package path to create the full module name.
        import_mod = f".{name}"
        full_import_mod = f"{parent_mod}{import_mod}"
    if fromlist:
        _log_event(
            f"{parent_mod}:{parent_mod_lineno} importing {fromlist} from {import_mod!r} "
            f"({cached=}, {level=})"
        )
    else:
        _log_event(
            f"{parent_mod}:{parent_mod_lineno} importing {import_mod!r} "
            f"({cached=}, {level=})"
        )
    start = time.time()
    try:
        return import_actual(name, globals, locals, fromlist, level)
    finally:
        if not cached:
            _log_event(
                f"Imported {full_import_mod!r} in {time.time() - start:.2f} seconds"
            )


def setup_logging():
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.FileHandler(LOG_FILE))
    logger.addHandler(logging.StreamHandler(stream=sys.stderr))


def register():
    setup_logging()
    mock.patch("builtins.__import__", import_mock).start()


@dataclass
class Import:
    name: str
    start_time_ms: int
    end_time_ms: int
    direct_imports: dict[int, Import] = dataclasses.field(default_factory=dict)

    @property
    def elapsed_ms(self) -> int:
        return self.end_time_ms - self.start_time_ms


def process_data():
    def handle_import(log_lines: Iterator[str]) -> Import:
        timestamp, line = next(log_lines).split(" -- ")
        match = re.fullmatch(
            r"([\w.]+):(\d+) importing '(\w+)' \(cached=(True|False), .*\)", line
        )
        assert match
        import_name = match[1]

    with open(LOG_FILE) as f:
        root_import = handle_import(f)

    return root_import


def parse_args():
    parser = argparse.ArgumentParser("impprof", allow_abbrev=False)
    group = parser.add_mutually_exclusive_group()
    group.required = True
    group.add_argument("-f", dest="file", nargs=argparse.REMAINDER, help="File to run")
    group.add_argument(
        "-m", dest="module", nargs=argparse.REMAINDER, help="Module to run"
    )
    return parser.parse_args()


def main():
    if len(sys.argv) <= 1:
        print("Expected file or module to run", file=sys.stderr)
        sys.exit(2)

    args = parse_args()

    register()

    if args.module:
        sys.argv = args.module
        runpy.run_module(sys.argv[0], run_name="__main__")
    else:
        sys.argv = args.file
        runpy.run_path(sys.argv[0], run_name="__main__")

    process_data()


if __name__ == "__main__":
    main()
