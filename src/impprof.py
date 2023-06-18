#!/usr/bin/env python3

__all__ = ("Import", "Monitor")

import argparse
import contextlib
import dataclasses
import importlib
import importlib.abc
import importlib.util
import inspect
import logging
import runpy
import sys
import textwrap
import time
import traceback
from collections import namedtuple
from dataclasses import dataclass
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import Container, List, Optional, Tuple, Dict
from unittest import mock


LOG_FILE = f".impprof.{int(time.time())}.log"

logger = logging.getLogger("impprof")

_import_actual = __import__
_sys_modules_actual = sys.modules


_ImportLoc = namedtuple("_ImportLoc", "package, lineno")


def _get_import_position() -> Optional[_ImportLoc]:
    """
    Try to get the module and line number of the frame triggering an import.

    :return:
        Module name and line number, if available, otherwise None.
    """
    frame = inspect.currentframe().f_back.f_back
    while frame:
        frame_info = inspect.getframeinfo(frame)
        frame_module = frame.f_globals["__name__"]
        # print(f"{frame_module}:{frame_info.lineno}")
        if frame_module.startswith("importlib"):
            frame = frame.f_back
        elif frame_module == "impprof":
            return None
        else:
            return _ImportLoc(frame_module, frame_info.lineno)
    return None


@dataclass
class Import:
    """Representation of a single import."""

    module: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    # Mapping with keys being the line number of the import.
    direct_imports: Dict[Tuple[str, int], "Import"] = dataclasses.field(
        default_factory=dict
    )

    def __str__(self):
        if self.direct_imports:
            direct_imports_str = "\n" + "\n".join(
                f"    line {i[1]}: {x.module}" for i, x in self.direct_imports.items()
            )
        else:
            direct_imports_str = "   <none>"
        return textwrap.dedent(
            f"""\
            Import: {self.module}
              elapsed:   {self.elapsed_ms} ms
              accounted: {self.accounted_ms} ms
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
        return self.elapsed_ms - sum(
            x.elapsed_ms for x in self.direct_imports.values() if x.end_time
        )


class TimingLoader(importlib.abc.Loader):
    def __init__(
        self,
        orig_loader: importlib.abc.Loader,
        imports: Dict[str, Import],
        mod_name: str,
        logger: logging.Logger,
    ):
        self._orig_loader = orig_loader
        self._imports = imports
        self._mod_name = mod_name
        self._logger = logger
        for method in ["create_module", "get_code"]:
            if hasattr(self._orig_loader, method):
                setattr(self, method, getattr(self._orig_loader, method))

    def exec_module(self, module: ModuleType) -> None:
        # Get or create the import object.
        if self._mod_name in self._imports:
            self._logger.warning(
                "Unexpectedly found %r already in imports", self._mod_name
            )
        import_obj = Import(self._mod_name)
        self._imports[self._mod_name] = import_obj

        # Add to the parent's 'direct_imports'.
        imp_loc = _get_import_position()
        if not imp_loc:
            incomplete_imports = [
                x.module
                for x in self._imports.values()
                if not x.end_time and x.module != self._mod_name
            ]
            if incomplete_imports:
                self._logger.warning(
                    "Unable to get parent module from stack, assigning to latest partial import %r",
                    incomplete_imports[-1],
                )
                imp_loc = _ImportLoc(incomplete_imports[-1], 0)
        if imp_loc and imp_loc.package in self._imports and imp_loc.lineno:
            self._imports[imp_loc.package].direct_imports[
                (self._mod_name, imp_loc.lineno)
            ] = import_obj
        elif imp_loc:
            self._logger.warning(
                "Unable to add %s to parent module %s", self._mod_name, imp_loc.package
            )
            # traceback.print_stack()
        else:
            self._logger.warning("Unable to find parent module for %s", self._mod_name)
            # traceback.print_stack()

        imp_loc_str = f"{imp_loc.package}:{imp_loc.lineno}" if imp_loc else "<none>"
        self._logger.debug("Executing module '%s' (%s)", self._mod_name, imp_loc_str)
        import_obj.start_time = time.time()
        self._orig_loader.exec_module(module)
        import_obj.end_time = time.time()
        self._logger.debug("Finished executing module '%s'", self._mod_name)


class TimingFinder(importlib.abc.MetaPathFinder):
    def __init__(
        self,
        imports: Dict[str, Import],
        logger: logging.Logger,
        ignore: Container[str] = (),
    ):
        self._imports = imports
        self._logger = logger
        self._ignore_pkgs = ignore
        # if "importlib" not in self._ignore_pkgs:
        #     self._ignore_pkgs = set(self._ignore_pkgs) | {"importlib"}

    def find_spec(
        self,
        fullname: str,
        path: Optional[str] = None,
        target: Optional[ModuleType] = None,
    ) -> Optional[ModuleSpec]:
        if any(fullname.partition(".")[0] == p for p in self._ignore_pkgs):
            return None
        with mock.patch(
            "sys.meta_path",
            [x for x in sys.meta_path if type(x) is not TimingFinder],
        ):
            spec = importlib.util.find_spec(fullname)
        if spec and hasattr(spec.loader, "exec_module"):
            spec.loader = TimingLoader(
                spec.loader, self._imports, fullname, self._logger
            )
        return spec


class Monitor:
    """Monitor used for tracking imports."""

    _active: bool = False
    _count: int = 0

    def __init__(self, ignore: Container[str] = ("sys", "_io", "os")):
        self._id = self.__class__._count
        self.__class__._count += 1
        self._ignore_pkgs = ignore
        self.imports: dict[str, Import] = {}
        self._logger = logger.getChild(f"Monitor-{self._id}")

    def __enter__(self) -> "Monitor":
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
        sys.meta_path.insert(
            0, TimingFinder(self.imports, self._logger, self._ignore_pkgs)
        )

    def stop(self) -> None:
        finder = sys.meta_path.pop(0)
        assert type(finder) == TimingFinder
        self.__class__._active = False
        self._logger.debug("Stopped import monitoring")


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

    formatter = ShortLevelNameFormatter(
        style="{",
        fmt="{asctime} {levelname:>5s} [{name:<13s}] - {message}",
    )
    logger.setLevel(logging.DEBUG)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    # fh = logging.FileHandler(LOG_FILE)
    # fh.setFormatter(formatter)
    # logger.addHandler(fh)


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser("impprof", allow_abbrev=False)
    group = parser.add_mutually_exclusive_group()
    group.required = True
    group.add_argument("-f", dest="file", nargs=argparse.REMAINDER, help="File to run")
    group.add_argument(
        "-m", dest="module", nargs=argparse.REMAINDER, help="Module to run"
    )
    group.add_argument("-i", dest="imp", help="Module to import")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> Monitor:
    args = parse_args(argv)
    setup_logging()

    with Monitor() as monitor:
        with contextlib.suppress(SystemExit):
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

    return monitor


if __name__ == "__main__":
    monitor = main()
