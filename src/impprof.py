#!/usr/bin/env python3

from __future__ import annotations


__all__ = ("Import", "Monitor")

import argparse
import dataclasses
import importlib
import inspect
import logging
import runpy
import sys
import textwrap
import time
from collections import namedtuple
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Container, Iterable, Mapping, Optional
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


class Monitor:

    _active: bool = False
    _count: int = 0

    _CallerInfo = namedtuple("_CallerInfo", "pkg, lineno")

    def __init__(self, ignore: Container[str] = ("sys", "_io")):
        self._id = self.__class__._count
        self.__class__._count += 1
        self._ignore_pkgs = ignore
        self.imports: dict[str, Import] = {}
        self._import_patch = mock.patch("builtins.__import__", self._import_mock)
        self._logger = logger.getChild(f"Monitor-{self._id}")

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
        self._logger.debug("Starting import monitoring")
        self._import_patch.start()

    def stop(self) -> None:
        self._import_patch.stop()
        self.__class__._active = False
        self._logger.debug("Stopped import monitoring")

    @classmethod
    def _get_calling_module(cls) -> _CallerInfo | None:
        """
        Try to get the module and line number of the calling frame.

        :return:
            Module name and line number, if available, otherwise None.
        """
        try:
            calling_frame = inspect.currentframe().f_back.f_back
            frame_info = inspect.getframeinfo(calling_frame)
            frame_module = calling_frame.f_globals["__name__"]
            if frame_module.startswith("importlib._bootstrap"):
                # TODO: What is this case?
                return None
            return cls._CallerInfo(frame_module, frame_info.lineno)
        except Exception:
            # TODO: Catch more specific case(s) (only one frame?)
            return None

    def _import_mock(
        self,
        name: str,
        globals: Mapping[str, Any] = None,
        locals: None = None,
        fromlist: Iterable[str] = (),
        level: int = 0,
    ) -> ModuleType:
        """
        Replacement for the builtin 'import' (builtins.__import__).

        Implements the required monitoring before forwarding to the real import
        implementation.

        See https://docs.python.org/3/library/functions.html#import__.

        When importing a module from a package, note that __import__('A.B', ...)
        returns package A when fromlist is empty, but its submodule B when
        fromlist is not empty.

        :param name:
            The name of the import.
        :param globals:
            Used to determine the context, not modified.
        :param locals:
            Ignored.
        :param fromlist:
            A list of names to emulate 'from <name> import ...', or empty to
            emulate 'import <name>'.
        :param level:
            Used to determine whether to perform absolute or relative imports:
            0 is absolute, while a positive number is the number of parent
            directories to search relative to the current module.
        :return:
            The imported module.
        """
        if any(name == p or name.startswith(f"{p}.") for p in self._ignore_pkgs):
            return _import_actual(name, globals, locals, fromlist, level)

        cached = name in sys.modules

        caller = self._get_calling_module()
        caller_str = f"{caller.pkg}:{caller.lineno}" if caller else "<none>"

        if level == 0:
            import_mod = name
            full_import_mod = name
        else:
            # Add the parent package path to create the full module name.
            import_mod = f".{name}"
            assert caller is not None
            full_import_mod = f"{caller.pkg}{import_mod}"
        if fromlist:
            self._logger.debug(
                "Handling %s 'from %s import %s' (cached=%s, level=%d)",
                caller_str,
                import_mod,
                ", ".join(fromlist),
                cached,
                level,
            )
        else:
            self._logger.debug(
                "Handling %s 'import %s' (cached=%s, level=%d)",
                caller_str,
                import_mod,
                cached,
                level,
            )
        start = time.time()
        try:
            imp_obj = self.imports[full_import_mod]
        except KeyError:
            imp_obj = Import(full_import_mod, start)
            self.imports[full_import_mod] = imp_obj
        if caller and caller.pkg in self.imports:
            self.imports[caller.pkg].direct_imports[caller.lineno] = imp_obj
        elif caller:
            self._logger.warning(
                "Unable to add %s to parent module %s", import_mod, caller_str
            )
        try:
            return _import_actual(name, globals, locals, fromlist, level)
        finally:
            if not cached:
                end = time.time()
                self._logger.debug("Imported %s", import_mod)
                imp_obj.end_time = end


def setup_logging() -> None:
    """Set up logging for CLI invocation."""

    class ShortLevelNameFormatter(logging.Formatter):
        """Log formatter, using log level names no longer than 5 characters."""

        def format(self, record: logging.LogRecord) -> str:
            if record.levelno == logging.WARNING:
                record.levelname = "WARN"
            elif record.levelno == logging.CRITICAL:
                record.levelname = "CRIT"
            return super().format(record)

    sh = logging.StreamHandler()
    fh = logging.FileHandler(LOG_FILE)
    formatter = ShortLevelNameFormatter(
        style="{",
        fmt="{asctime} {levelname:>5s} [{name:<13s}] - {message}",
    )
    sh.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser("impprof", allow_abbrev=False)
    group = parser.add_mutually_exclusive_group()
    group.required = True
    group.add_argument("-f", dest="file", nargs=argparse.REMAINDER, help="File to run")
    group.add_argument(
        "-m", dest="module", nargs=argparse.REMAINDER, help="Module to run"
    )
    group.add_argument("-i", dest="imp", help="Module to import")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    setup_logging()

    with Monitor() as monitor:
        if args.module:
            with mock.patch("sys.argv", args.module):
                runpy.run_module(sys.argv[0], run_name="__main__")
        elif args.file:
            with mock.patch("sys.argv", args.file):
                runpy.run_path(sys.argv[0], run_name="__main__")
        elif args.imp:
            importlib.import_module(args.imp)
        else:
            assert False


if __name__ == "__main__":
    main()
