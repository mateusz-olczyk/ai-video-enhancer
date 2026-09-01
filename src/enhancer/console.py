"""Shared Rich console and progress UI for CLI and pipeline."""

import sys

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

stdout = Console(highlight=False)
stderr = Console(highlight=False, file=sys.stderr)


def make_pipeline_progress() -> Progress:
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[progress.percentage]{task.percentage:>5.1f}%"),
        TimeElapsedColumn(),
        TextColumn("<"),
        TimeRemainingColumn(),
        TextColumn("{task.fields[speed]:.2f} frame/s"),
        console=stdout,
        transient=False,
    )
