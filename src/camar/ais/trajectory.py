from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class AISTrajectory:
    """A single continuous vessel track extracted from AIS data.

    All arrays are 1-D with the same length ``n_points``.

    Attributes
    ----------
    mmsi:
        Vessel identifier.
    timestamps:
        UTC Unix timestamps in seconds (float64).
    lons, lats:
        WGS84 longitude/latitude in degrees (float64).
    speeds_kn:
        Speed over ground in knots (float64).
    courses_deg:
        Course over ground in degrees 0–360 (float64).
    headings_deg:
        Magnetic heading in degrees 0–360 (float64).  Defaults to zeros if
        the source data does not contain a heading field.
    ship_type:
        AIS cargo ship type code.
    """

    mmsi: int
    timestamps: np.ndarray    # float64, Unix seconds
    lons: np.ndarray          # float64, degrees
    lats: np.ndarray          # float64, degrees
    speeds_kn: np.ndarray     # float64, knots
    courses_deg: np.ndarray   # float64, degrees
    headings_deg: np.ndarray = None  # float64, degrees (magnetic heading)
    ship_type: int = 0

    def __post_init__(self):
        if self.headings_deg is None:
            self.headings_deg = np.zeros(len(self.timestamps), dtype=np.float64)

    @property
    def n_points(self) -> int:
        return len(self.timestamps)

    @property
    def duration_s(self) -> float:
        return float(self.timestamps[-1] - self.timestamps[0])

    def __repr__(self) -> str:
        return (
            f"AISTrajectory(mmsi={self.mmsi}, n_points={self.n_points}, "
            f"duration_s={self.duration_s:.0f}, ship_type={self.ship_type})"
        )


def extract_trajectories(
    df: pd.DataFrame,
    min_points: int = 10,
    max_gap_s: float = 300.0,
) -> list[AISTrajectory]:
    """Segment AIS DataFrame into individual vessel tracks.

    Splits a vessel's track wherever the gap between consecutive AIS messages
    exceeds ``max_gap_s`` seconds, then discards segments shorter than
    ``min_points`` rows.

    Parameters
    ----------
    df:
        DataFrame as returned by :func:`load_ais_parquet`.
    min_points:
        Minimum number of points a segment must have to be kept.
    max_gap_s:
        Maximum allowed gap in seconds between consecutive messages before
        a new segment is started.

    Returns
    -------
    list[AISTrajectory]
        Flat list of trajectory segments, sorted by (mmsi, start_time).
    """
    trajectories: list[AISTrajectory] = []

    for mmsi, group in df.groupby("ais_static_mmsi", sort=False):
        group = group.sort_values("ais_timestamp").drop_duplicates(subset="ais_timestamp")

        ts = group["ais_timestamp"].values.astype("datetime64[s]").astype(np.float64)
        lons = group["longitude_degrees"].values
        lats = group["latitude_degrees"].values
        speeds = group["speed"].values
        courses = group["course"].values
        headings = group["heading"].values.astype(np.float64) if "heading" in group.columns else np.zeros(len(ts))
        ship_type = int(group["ais_cargo_ship_type"].iloc[0])

        # Find split points where time gap exceeds max_gap_s
        gaps = np.diff(ts)
        split_indices = np.where(gaps > max_gap_s)[0] + 1  # indices of segment starts

        segment_starts = np.concatenate([[0], split_indices])
        segment_ends = np.concatenate([split_indices, [len(ts)]])

        for start, end in zip(segment_starts, segment_ends):
            if end - start < min_points:
                continue
            trajectories.append(
                AISTrajectory(
                    mmsi=int(mmsi),
                    timestamps=ts[start:end].copy(),
                    lons=lons[start:end].copy(),
                    lats=lats[start:end].copy(),
                    speeds_kn=speeds[start:end].copy(),
                    courses_deg=courses[start:end].copy(),
                    headings_deg=headings[start:end].copy(),
                    ship_type=ship_type,
                )
            )

    trajectories.sort(key=lambda t: (t.mmsi, t.timestamps[0]))
    return trajectories
