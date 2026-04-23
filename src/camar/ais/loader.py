from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd


def load_ais_parquet(
    paths: Union[str, Path, list[Union[str, Path]]],
    bbox: Optional[tuple[float, float, float, float]] = None,
    time_range: Optional[tuple[str, str]] = None,
    mmsi_filter: Optional[list[int]] = None,
) -> pd.DataFrame:
    """Load AIS data from one or more parquet files with optional filtering.

    Parameters
    ----------
    paths:
        Path or list of paths to ``.parquet`` files.
    bbox:
        ``(lon_min, lat_min, lon_max, lat_max)`` in WGS84 degrees.
        Rows outside this box are dropped.
    time_range:
        ``(start, end)`` as ISO-8601 strings, e.g. ``("2018-12-29", "2018-12-29 10:00")``.
        Both ends are inclusive.
    mmsi_filter:
        If provided, only keep rows whose ``ais_static_mmsi`` is in this list.

    Returns
    -------
    pd.DataFrame
        Sorted by ``(ais_static_mmsi, ais_timestamp)``, index reset.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]

    frames = [pd.read_parquet(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)

    if bbox is not None:
        lon_min, lat_min, lon_max, lat_max = bbox
        df = df[
            (df["longitude_degrees"] >= lon_min)
            & (df["longitude_degrees"] <= lon_max)
            & (df["latitude_degrees"] >= lat_min)
            & (df["latitude_degrees"] <= lat_max)
        ]

    if time_range is not None:
        t_start, t_end = pd.Timestamp(time_range[0]), pd.Timestamp(time_range[1])
        df = df[(df["ais_timestamp"] >= t_start) & (df["ais_timestamp"] <= t_end)]

    if mmsi_filter is not None:
        df = df[df["ais_static_mmsi"].isin(mmsi_filter)]

    df = df.sort_values(["ais_static_mmsi", "ais_timestamp"]).reset_index(drop=True)
    return df
