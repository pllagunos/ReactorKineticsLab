"""OpenMC rod-worth reference loading and interpolation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROD_WORTH_CSV = (
    _REPOSITORY_ROOT
    / "openmc"
    / "reference_data"
    / "concentric"
    / "rod_scan"
    / "results"
    / "rod_worth.csv"
)


@dataclass(frozen=True)
class RodWorthTable:
    insertion_fraction: tuple[float, ...]
    rho_total_pcm: tuple[float, ...]
    rho_rod_worth_pcm: tuple[float, ...]

    @property
    def clean_excess_pcm(self) -> float:
        return self.rho_total_pcm[0]

    @property
    def full_insertion_worth_pcm(self) -> float:
        return abs(self.rho_rod_worth_pcm[-1])

    @property
    def critical_insertion_percent(self) -> float:
        xs = self.insertion_fraction
        ys = self.rho_total_pcm
        for index in range(len(xs) - 1):
            y0 = ys[index]
            y1 = ys[index + 1]
            if y0 == 0.0:
                return 100.0 * xs[index]
            if y0 * y1 <= 0.0 and y1 != y0:
                x0 = xs[index]
                x1 = xs[index + 1]
                return 100.0 * (x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0))
        if ys[-1] == 0.0:
            return 100.0 * xs[-1]
        return 100.0 * xs[-1]

    def interpolate_rod_worth_pcm(self, insertion_fraction: float) -> float:
        return _interpolate(
            self.insertion_fraction,
            self.rho_rod_worth_pcm,
            insertion_fraction,
        )

    def interpolate_total_pcm(self, insertion_fraction: float) -> float:
        return _interpolate(
            self.insertion_fraction,
            self.rho_total_pcm,
            insertion_fraction,
        )


def _interpolate(
    xs: tuple[float, ...],
    ys: tuple[float, ...],
    x: float,
) -> float:
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for index in range(len(xs) - 1):
        x0 = xs[index]
        x1 = xs[index + 1]
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0)
            return ys[index] + t * (ys[index + 1] - ys[index])
    return ys[-1]


def load_rod_worth_table(path: str | Path = DEFAULT_ROD_WORTH_CSV) -> RodWorthTable:
    table_path = Path(path)
    rows: list[tuple[float, float, float]] = []
    with table_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"insertion_fraction", "rho_total_pcm", "rho_rod_worth_pcm"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                "rod-worth CSV must contain insertion_fraction, "
                "rho_total_pcm, and rho_rod_worth_pcm"
            )
        for row in reader:
            rows.append(
                (
                    float(row["insertion_fraction"]),
                    float(row["rho_total_pcm"]),
                    float(row["rho_rod_worth_pcm"]),
                )
            )
    if len(rows) < 2:
        raise ValueError("rod-worth CSV must contain at least two rows")
    xs = tuple(row[0] for row in rows)
    if xs[0] != 0.0 or xs[-1] != 1.0:
        raise ValueError("rod-worth CSV must span insertion fractions 0 to 1")
    if any(b <= a for a, b in zip(xs[:-1], xs[1:])):
        raise ValueError("rod-worth CSV insertion fractions must increase")
    worth = tuple(row[2] for row in rows)
    if worth[0] != 0.0:
        raise ValueError("rod-worth CSV must define zero worth at x=0")
    return RodWorthTable(
        insertion_fraction=xs,
        rho_total_pcm=tuple(row[1] for row in rows),
        rho_rod_worth_pcm=worth,
    )


ROD_WORTH_TABLE = load_rod_worth_table()
