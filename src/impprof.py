#!/usr/bin/env python3

from __future__ import annotations

__all__ = ("Import", "Monitor")

import argparse
import dataclasses
import inspect
import logging
import runpy
import sys
import textwrap
import time
from dataclasses import dataclass
from types import ModuleType
from typing import Container, Optional
from unittest import mock


LOG_FILE = f".impprof.{int(time.time())}.log"

logger = logging.getLogger("impprof")

_import_actual = __import__

_sys_modules_actual = sys.modules


@dataclass
class Import:
    module: str
    start_time: float
    end_time: Optional[float] = None
    # Mapping with keys being the line number of the import.
    direct_imports: dict[int, Import] = dataclasses.field(default_factory=dict)

    def __str__(self):
        if self.direct_imports:
            direct_imports_str = "\n" + "\n".join(
                f"    line {i}: {x.module}" for i, x in self.direct_imports.items()
            )
        else:
            direct_imports_str = "   <none>"
        return textwrap.dedent(
            f"""\
            Import: {self.module}
              elapsed:   {self.elapsed_ms} ms
              accounted: {self.elapsed_ms} ms
              imports:{{}}
            """
        ).format(direct_imports_str)

    @property
    def elapsed_ms(self) -> Optional[int]:
        if self.end_time is None:
            return None
        return int(1000 * (self.end_time - self.start_time))

    @property
    def accounted_ms(self) -> Optional[int]:
        if self.end_time is None:
            return None
        return self.elapsed_ms - sum(x.elapsed_ms for x in self.direct_imports.values())


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


class _SysModulesMock(dict):
    """
    Mock replacement for sys.modules.

    In the case of setting an item which is not already in sys.modules, the item
    will also be added into sys.modules.
    """

    def __setitem__(self, key: str, value: ModuleType) -> None:
        super().__setitem__(key, value)
        # Add to real sys.modules only if it wasn't already in there!
        if key not in _sys_modules_actual:
            _sys_modules_actual[key] = value


class Monitor:

    _active: bool = False

    def __init__(self, ignore: Container[str] = ("sys", "_io")):
        self._ignore_pkgs = ignore
        self.imports: dict[str, Import] = {}
        self._import_patch = mock.patch("builtins.__import__", self._import_mock)
        self._sys_modules_mock = _SysModulesMock()
        self._sys_modules_patch = mock.patch("sys.modules", self._sys_modules_mock)

    def __enter__(self) -> Monitor:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    def start(self) -> None:
        if self._active:
            raise RuntimeError("Unable to start import monitor - already started")
        self.__class__._active = True
        logger.debug("Starting import monitoring")
        self._import_patch.start()
        self._sys_modules_patch.start()

    def stop(self) -> None:
        self._import_patch.stop()
        self._sys_modules_patch.stop()
        self.__class__._active = False
        logger.debug("Stopped import monitoring")

    def _import_mock(
        self,
        name: str,
        globals=None,
        locals=None,
        fromlist=(),
        level: int = 0,
    ) -> ModuleType:
        if any(name == p or name.startswith(f"{p}.") for p in self._ignore_pkgs):
            return _import_actual(name, globals, locals, fromlist, level)

        cached = name in sys.modules

        parent_mod, parent_mod_lineno = _get_calling_module()
        if parent_mod.startswith("importlib._bootstrap"):
            return _import_actual(name, globals, locals, fromlist, level)

        if level == 0:
            import_mod = name
            full_import_mod = name
        else:
            # Add the parent package path to create the full module name.
            import_mod = f".{name}"
            full_import_mod = f"{parent_mod}{import_mod}"
        if fromlist:
            logger.debug(
                "Handling %s:%d 'from %s import %s' (cached=%s)",
                parent_mod,
                parent_mod_lineno,
                import_mod,
                ", ".join(fromlist),
                cached,
            )
            _log_event(
                f"{parent_mod}:{parent_mod_lineno} importing {fromlist} from {import_mod!r} "
                f"({cached=}, {level=})"
            )
        else:
            logger.debug(
                "Handling %s:%d 'import %s' (cached=%s)",
                parent_mod,
                parent_mod_lineno,
                import_mod,
                cached,
            )
            _log_event(
                f"{parent_mod}:{parent_mod_lineno} importing {import_mod!r} "
                f"({cached=}, {level=})"
            )
        start = time.time()
        try:
            imp_obj = self.imports[full_import_mod]
        except KeyError:
            imp_obj = Import(full_import_mod, start)
            self.imports[full_import_mod] = imp_obj
        if parent_mod in self.imports and parent_mod_lineno:
            self.imports[parent_mod].direct_imports[parent_mod_lineno] = imp_obj
        else:
            logger.warning(
                "Unable to add %s to parent module %s:%d",
                import_mod,
                parent_mod,
                parent_mod_lineno,
            )
        try:
            return _import_actual(name, globals, locals, fromlist, level)
        finally:
            if not cached:
                end = time.time()
                _log_event(f"Imported {full_import_mod!r} in {end-start:.2f} seconds")
                imp_obj.end_time = end


def setup_logging():
    logging.basicConfig(format="%(asctime)s [%(levelname)s:%(name)s] - %(message)s")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.FileHandler(LOG_FILE))
    # logger.addHandler(logging.StreamHandler(stream=sys.stderr))


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser("impprof", allow_abbrev=False)
    group = parser.add_mutually_exclusive_group()
    group.required = True
    group.add_argument("-f", dest="file", nargs=argparse.REMAINDER, help="File to run")
    group.add_argument(
        "-m", dest="module", nargs=argparse.REMAINDER, help="Module to run"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    setup_logging()

    with Monitor() as monitor:
        if args.module:
            with mock.patch("sys.argv", args.module):
                runpy.run_module(sys.argv[0], run_name="__main__")
        else:
            with mock.patch("sys.argv", args.file):
                runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
