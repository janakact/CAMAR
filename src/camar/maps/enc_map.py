import json
import math
import os
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .base import base_map
from .const import ENV_DEVICE
from camar.registry import register_map


class ENCProjection:
    """
    Equirectangular projection between WGS84 lon/lat (degrees) and
    flat Euclidean km coordinates centred at (lon_c, lat_c).

    Convention
    ----------
    x  = (lon - lon_c) * cos(lat_c) * 111.32   [km, east positive]
    y  = (lat_c - lat) * 111.32                 [km, north = negative y]

    The y sign is chosen so that the SVG coordinate system (y increasing
    downward) shows north at the top without any extra flip.  The matplotlib
    renderer already inverts y, so north appears at the top there too.

    Can be applied to any lon/lat data (e.g. AIS trajectories) that share
    the same reference point.
    """

    METERS_PER_DEGREE = 111_320.0  # metres per degree of latitude
    KM_PER_DEGREE = 111.32         # km per degree of latitude

    def __init__(self, lon_c: float, lat_c: float):
        self.lon_c = lon_c
        self.lat_c = lat_c
        self._cos_lat = math.cos(math.radians(lat_c))

    # ------------------------------------------------------------------
    # scalar / numpy API
    # ------------------------------------------------------------------

    def forward(self, lon, lat):
        """(lon, lat) -> (x_km, y_km).  Accepts scalars or numpy arrays."""
        x = (np.asarray(lon) - self.lon_c) * self._cos_lat * self.KM_PER_DEGREE
        y = (self.lat_c - np.asarray(lat)) * self.KM_PER_DEGREE
        return x, y

    def inverse(self, x, y):
        """(x_km, y_km) -> (lon, lat).  Accepts scalars or numpy arrays."""
        lon = np.asarray(x) / (self._cos_lat * self.KM_PER_DEGREE) + self.lon_c
        lat = self.lat_c - np.asarray(y) / self.KM_PER_DEGREE
        return lon, lat

    # ------------------------------------------------------------------
    # JAX API (for use inside JIT-compiled code)
    # ------------------------------------------------------------------

    def forward_jax(self, lon, lat):
        """(lon, lat) -> (x_km, y_km) using JAX arrays."""
        x = (lon - self.lon_c) * self._cos_lat * self.KM_PER_DEGREE
        y = (self.lat_c - lat) * self.KM_PER_DEGREE
        return x, y

    def inverse_jax(self, x, y):
        """(x_km, y_km) -> (lon, lat) using JAX arrays."""
        lon = x / (self._cos_lat * self.KM_PER_DEGREE) + self.lon_c
        lat = self.lat_c - y / self.KM_PER_DEGREE
        return lon, lat

    def __repr__(self):
        return f"ENCProjection(lon_c={self.lon_c}, lat_c={self.lat_c})"


def _project_polygon_exterior(geom, proj: ENCProjection):
    """Return projected exterior ring of a shapely geometry as (N, 2) numpy array."""
    from shapely.geometry import MultiPolygon, Polygon

    rings = []
    if isinstance(geom, Polygon):
        polys = [geom]
    elif isinstance(geom, MultiPolygon):
        polys = list(geom.geoms)
    else:
        return np.empty((0, 2), dtype=np.float64)

    for poly in polys:
        coords = np.array(poly.exterior.coords)  # (N, 2) lon/lat
        x, y = proj.forward(coords[:, 0], coords[:, 1])
        rings.append(np.stack([x, y], axis=1))

    return rings


def _sample_linestring(linestring, spacing_km: float):
    """Uniformly sample points along a shapely LineString every spacing_km km."""
    length = linestring.length  # already in km after projection
    if length == 0:
        return []
    n = max(1, int(math.ceil(length / spacing_km)))
    distances = np.linspace(0, length, n, endpoint=False)
    pts = [linestring.interpolate(d) for d in distances]
    return [(p.x, p.y) for p in pts]


@register_map("enc_map")
class enc_map(base_map):
    """
    Maritime map loaded from ENC (Electronic Navigation Chart) shapefiles.

    Land polygon boundaries are sampled at regular intervals and converted to
    circular landmarks.  Anchorage and TSS zones are stored for rendering only
    (they do not affect physics or observations).

    Parameters
    ----------
    enc_dir : str
        Directory containing ``landPolygons.shp``, ``zones.shp``, and
        ``labels.json``.
    num_agents : int
        Number of agents.
    bbox : tuple (lon_min, lat_min, lon_max, lat_max), optional
        Bounding box in WGS84 degrees.  If *None* the bbox is derived
        automatically from ``landPolygons.shp``.
    coastline_sampling_km : float
        Distance between consecutive coastline-sample landmark circles (km).
        Landmark radius is set to half this value.
    agent_rad_km : float
        Agent radius in km.
    goal_rad_km : float
        Goal-zone radius in km.
    free_pos_spacing_km : float
        Grid spacing (km) used when sampling valid water positions for
        agent / goal placement.
    """

    def __init__(
        self,
        enc_dir: str,
        num_agents: int = 8,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        coastline_sampling_km: float = 0.5,
        agent_rad_km: float = 0.15,
        goal_rad_km: float = 0.06,
        free_pos_spacing_km: float = 0.5,
    ):
        try:
            import geopandas as gpd
            from shapely.geometry import LineString, MultiPolygon, Point, Polygon
            from shapely.ops import unary_union
        except ImportError as e:
            raise ImportError(
                "enc_map requires geopandas and shapely. "
                "Install them with: mamba install -c conda-forge geopandas shapely"
            ) from e

        self._num_agents = num_agents
        self._coastline_sampling_km = coastline_sampling_km
        self._agent_rad_km = agent_rad_km
        self._goal_rad_km = goal_rad_km

        # ------------------------------------------------------------------
        # 1. Load land polygons
        # ------------------------------------------------------------------
        land_path = os.path.join(enc_dir, "landPolygons.shp")
        land_gdf = gpd.read_file(land_path)

        # Resolve bounding box
        if bbox is not None:
            lon_min, lat_min, lon_max, lat_max = bbox
        else:
            lon_min, lat_min, lon_max, lat_max = land_gdf.total_bounds  # (minx, miny, maxx, maxy)

        lon_c = (lon_min + lon_max) / 2.0
        lat_c = (lat_min + lat_max) / 2.0

        self.projection = ENCProjection(lon_c, lat_c)

        # Map dimensions in km
        x_min, y_min = self.projection.forward(lon_min, lat_max)  # NW corner
        x_max, y_max = self.projection.forward(lon_max, lat_min)  # SE corner
        self._width = float(x_max - x_min)
        self._height = float(y_max - y_min)

        # ------------------------------------------------------------------
        # 2. Build projected land union (for water-mask tests)
        # ------------------------------------------------------------------
        projected_polys = []
        for geom in land_gdf.geometry:
            if geom is None:
                continue
            rings = _project_polygon_exterior(geom, self.projection)
            for ring_xy in rings:
                if len(ring_xy) >= 3:
                    projected_polys.append(Polygon(ring_xy))

        land_union = unary_union(projected_polys)

        # ------------------------------------------------------------------
        # 3. Sample coastlines → circular landmarks
        # ------------------------------------------------------------------
        landmark_pts = []
        for poly in projected_polys:
            exterior_line = LineString(poly.exterior.coords)
            pts = _sample_linestring(exterior_line, coastline_sampling_km)
            landmark_pts.extend(pts)

        self._landmark_pos = np.array(landmark_pts, dtype=np.float32)  # (N, 2)
        print(f"enc_map: {len(landmark_pts)} coastline landmark circles "
              f"(sampling={coastline_sampling_km} km, "
              f"radius={coastline_sampling_km / 2} km)")

        # ------------------------------------------------------------------
        # 4. Build free water positions (regular grid, excluding land)
        # ------------------------------------------------------------------
        xs = np.arange(x_min + free_pos_spacing_km / 2, x_max, free_pos_spacing_km)
        ys = np.arange(y_min + free_pos_spacing_km / 2, y_max, free_pos_spacing_km)
        xx, yy = np.meshgrid(xs, ys)
        grid_pts = np.stack([xx.ravel(), yy.ravel()], axis=1)

        # Vectorised containment check using shapely
        from shapely.vectorized import contains
        mask = ~contains(land_union, grid_pts[:, 0], grid_pts[:, 1])
        free_pts = grid_pts[mask]
        assert len(free_pts) >= num_agents, (
            f"Only {len(free_pts)} water positions found; need at least {num_agents}. "
            "Try a smaller free_pos_spacing_km or adjust bbox."
        )
        print(f"enc_map: {len(free_pts)} valid water positions "
              f"(grid spacing={free_pos_spacing_km} km)")

        # ------------------------------------------------------------------
        # 5. Load and project zones
        # ------------------------------------------------------------------
        zones_path = os.path.join(enc_dir, "zones.shp")
        labels_path = os.path.join(enc_dir, "labels.json")

        with open(labels_path) as f:
            labels = json.load(f)

        zones_gdf = gpd.read_file(zones_path)
        self.zones = []

        for _, row in zones_gdf.iterrows():
            zone_id = str(row.get("id", "")).strip()
            if zone_id not in labels:
                continue
            label = labels[zone_id]
            zone_type = label.get("type", "unknown")
            direction = label.get("direction", None)

            geom = row.geometry
            if geom is None:
                continue

            rings = _project_polygon_exterior(geom, self.projection)
            if not rings:
                continue
            ring_xy = rings[0]  # use exterior ring only

            # Centroid in projected coords
            projected_poly = Polygon(ring_xy)
            cx = projected_poly.centroid.x
            cy = projected_poly.centroid.y

            # Arrow length: ~half the "radius" of the polygon
            area = projected_poly.area
            arrow_len = min(math.sqrt(area / math.pi) * 0.5, 5.0)

            self.zones.append({
                "polygon": [(float(x), float(y)) for x, y in ring_xy],
                "type": zone_type,
                "direction": float(direction) if direction is not None else None,
                "centroid": (float(cx), float(cy)),
                "arrow_len": float(arrow_len),
            })

        print(f"enc_map: {len(self.zones)} zones loaded "
              f"({sum(1 for z in self.zones if z['type']=='anchorage')} anchorage, "
              f"{sum(1 for z in self.zones if z['type']=='tss')} TSS)")

        # ------------------------------------------------------------------
        # 6. Move arrays to JAX device
        # ------------------------------------------------------------------
        self._landmark_pos_jax = jnp.array(self._landmark_pos, device=ENV_DEVICE)
        self._free_pos_jax = jnp.array(free_pts, dtype=jnp.float32, device=ENV_DEVICE)

        self.setup_rad()

    # ------------------------------------------------------------------
    # base_map interface
    # ------------------------------------------------------------------

    def setup_rad(self):
        self.landmark_rad = self._coastline_sampling_km / 2.0
        self.agent_rad = self._agent_rad_km
        self.goal_rad = self._goal_rad_km
        self.proportional_goal_rad = False
        self.agent_rad_range = None
        self.landmark_rad_range = None
        self.goal_rad_range = None

    @property
    def homogeneous_agents(self) -> bool:
        return True

    @property
    def homogeneous_landmarks(self) -> bool:
        return True

    @property
    def homogeneous_goals(self) -> bool:
        return True

    @property
    def num_agents(self) -> int:
        return self._num_agents

    @property
    def num_landmarks(self) -> int:
        return int(self._landmark_pos_jax.shape[0])

    @property
    def height(self) -> float:
        return self._height

    @property
    def width(self) -> float:
        return self._width

    def reset(self, key: ArrayLike) -> tuple[Array, Array, Array, Array]:
        key, key_a, key_g = jax.random.split(key, 3)
        agent_pos = jax.random.choice(key_a, self._free_pos_jax, shape=(self.num_agents,), replace=False)
        goal_pos = jax.random.choice(key_g, self._free_pos_jax, shape=(self.num_agents,), replace=False)
        sizes = self.generate_sizes(key)
        return key_g, self._landmark_pos_jax, agent_pos, goal_pos, sizes
