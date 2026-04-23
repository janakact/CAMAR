"""
Generate a CAMAR maritime-simulator architecture diagram as PNG.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C = {
    "data":    "#3A7FC1",
    "enc":     "#27AE60",
    "dyn":     "#D68910",
    "env":     "#7D3C98",
    "policy":  "#C0392B",
    "render":  "#0E8080",
    "file":    "#5D6D7E",
    "bg":      "#12121E",
    "section": "#1E1E30",
    "sim":     "#1C2657",
}

# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------
FW, FH = 26, 17
fig, ax = plt.subplots(figsize=(FW, FH))
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis("off")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def box(ax, cx, cy, w, h, color, line1, line2="", fontsize=9, bold=False, radius=0.22):
    patch = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle=f"round,pad=0.04,rounding_size={radius}",
        facecolor=color, edgecolor="#FFFFFF", linewidth=0.5,
        alpha=0.95, zorder=3,
    )
    ax.add_patch(patch)
    weight = "bold" if bold else "normal"
    if line2:
        ax.text(cx, cy + 0.17, line1, ha="center", va="center",
                fontsize=fontsize, color="white", fontweight=weight, zorder=4)
        ax.text(cx, cy - 0.20, line2, ha="center", va="center",
                fontsize=fontsize - 1.5, color="#DDDDDD", style="italic", zorder=4)
    else:
        ax.text(cx, cy, line1, ha="center", va="center",
                fontsize=fontsize, color="white", fontweight=weight, zorder=4)


def section_bg(ax, x, y, w, h, title, color, title_color="#AAAAAA"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.3",
        facecolor=color, edgecolor="#444444", linewidth=0.8,
        alpha=0.35, zorder=1,
    )
    ax.add_patch(patch)
    ax.text(x + 0.18, y + h - 0.25, title,
            ha="left", va="top", fontsize=8.5,
            color=title_color, fontweight="bold", style="italic", zorder=2)


def arr(ax, x0, y0, x1, y1, label="", lw=1.3, color="#BBBBBB",
        conn="arc3,rad=0.0", label_side="top"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, connectionstyle=conn), zorder=5)
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        dy = 0.18 if label_side == "top" else -0.18
        ax.text(mx, my + dy, label, ha="center", va="center",
                fontsize=6.8, color="#EEEEEE", zorder=6,
                bbox=dict(facecolor=C["bg"], edgecolor="none", alpha=0.75, pad=1.5))


# ===========================================================================
# SECTION BACKGROUNDS
# ===========================================================================
section_bg(ax,  0.3, 9.0,  6.0,  7.8, "① AIS Data Pipeline",   "#1A3A5C")
section_bg(ax,  6.6, 11.5, 5.8,  5.2, "② ENC Map",              "#1A3D2A")
section_bg(ax,  6.6,  9.0, 5.8,  2.2, "③ Dynamics",             "#3D2000")
section_bg(ax, 12.7,  9.0, 7.0,  7.8, "④ CAMAR Environment",   "#2D1A4D")
section_bg(ax,  0.3,  0.5, 12.1, 8.3, "⑤ AIS Replay Policy",   "#4D0A0A")
section_bg(ax, 12.7,  0.5, 13.0, 8.3, "⑥ Rendering",            "#0A3030")

# ===========================================================================
# ① AIS DATA PIPELINE
# ===========================================================================
box(ax, 3.3, 16.3, 5.0, 0.70, C["file"],  "AIS Parquet Files",   bold=True, fontsize=9.5)
box(ax, 3.3, 15.2, 5.2, 0.72, C["data"],  "load_ais_parquet()", "bbox · time_range · mmsi_filter")
box(ax, 3.3, 14.1, 5.2, 0.70, C["data"],  "extract_trajectories()", "gap-split · min_points filter")
box(ax, 3.3, 13.0, 5.2, 0.72, C["data"],  "List[AISTrajectory]  (raw)", "irregular Δt · per-vessel t₀")
box(ax, 3.3, 11.85,5.2, 0.78, C["data"],  "align_trajectories_to_window()",
    "shared t_grid · slide window · clamp endpoints")
box(ax, 3.3, 10.65,5.2, 0.72, C["data"],  "List[AISTrajectory]  (aligned)",
    "same timestamps · speeds=0 outside range")

arr(ax, 3.3, 15.95, 3.3, 15.57, "DataFrame")
arr(ax, 3.3, 14.85, 3.3, 14.47)
arr(ax, 3.3, 13.75, 3.3, 13.37, "List[AISTrajectory]")
arr(ax, 3.3, 12.62, 3.3, 12.25, "raw trajs")
arr(ax, 3.3, 11.47, 3.3, 11.02, "(t_start, t_end)")

# ===========================================================================
# ② ENC MAP
# ===========================================================================
box(ax, 9.5, 16.3,  5.0, 0.70, C["file"],  "ENC Shapefiles + labels.json", bold=True, fontsize=9.5)
box(ax, 9.5, 15.2,  5.2, 0.72, C["enc"],   "ENCProjection",
    "lon/lat → km  (equirectangular)")
box(ax, 9.5, 14.0,  5.2, 0.80, C["enc"],   "enc_map",
    "landmarks · free-water grid · zones")
box(ax, 9.5, 12.85, 5.2, 0.65, C["enc"],   "set_fixed_positions(starts, goals)", fontsize=8.5)

arr(ax, 9.5, 15.95, 9.5, 15.57, "shapefiles")
arr(ax, 9.5, 14.85, 9.5, 14.42, "projection")
arr(ax, 9.5, 13.60, 9.5, 13.18)

# ③ DYNAMICS
box(ax, 9.5, 10.65, 5.2, 0.72, C["dyn"],   "DeltaPosDynamic",
    "max_speed · dt · max_angle_delta  →  action_size=3")
box(ax, 9.5,  9.60, 5.2, 0.72, C["dyn"],   "DeltaPosState",
    "agent_pos (N,2) · agent_angle (N,)  radians 0=north CW")

arr(ax, 9.5, 10.29, 9.5, 9.97)

# ===========================================================================
# ④ CAMAR ENVIRONMENT
# ===========================================================================
box(ax, 16.2, 16.3,  5.8, 0.70, C["env"],  "camar_v0(map, dynamic)", "make_env() factory",
    bold=True, fontsize=9.5)
box(ax, 16.2, 15.1,  5.8, 0.80, C["env"],  "Camar",
    "num_agents · height · width · step_dt · action_size", bold=True, fontsize=9.5)

# reset / step pair
box(ax, 14.6, 12.8,  2.2, 0.65, C["env"],  "reset(key)")
box(ax, 17.8, 12.8,  2.2, 0.65, C["env"],  "step(key, s, a)")

# State
box(ax, 16.2, 11.7,  5.8, 0.70, C["env"],  "State",
    "physical_state · landmark_pos · goal_pos · step")

# Sub-calls
box(ax, 14.6, 10.55, 2.4, 0.60, C["enc"],  "map.reset()", fontsize=8)
box(ax, 17.8, 10.55, 2.4, 0.60, C["dyn"],  "dynamic.integrate()", fontsize=8)
box(ax, 16.2,  9.60, 5.8, 0.65, C["env"],  "collision detection + observations + reward", fontsize=8.5)

arr(ax, 16.2, 15.95, 16.2, 15.52)
arr(ax, 14.6, 14.72, 14.6, 13.13)
arr(ax, 17.8, 14.72, 17.8, 13.13)
arr(ax, 14.6, 12.47, 14.6, 10.85)
arr(ax, 17.8, 12.47, 17.8, 10.85)
arr(ax, 14.6, 10.25, 16.2, 11.35, "pos/landmark")
arr(ax, 17.8, 10.25, 16.2, 11.35, "new pos")
arr(ax, 16.2, 11.35, 16.2,  9.93, "obs · state")

# enc_map → Camar
arr(ax, 12.1, 14.0, 12.7, 15.1, "map", lw=1.6, color="#27AE60")
# dyn → Camar
arr(ax, 12.1, 10.65, 14.3, 15.1, "dynamic", lw=1.6, color="#D68910",
    conn="arc3,rad=-0.25")

# ===========================================================================
# ⑤ AIS REPLAY POLICY
# ===========================================================================
# Inputs row
box(ax, 2.5, 8.1,  3.6, 0.68, C["data"],  "aligned trajs",
    "timestamps · lons · lats · headings")
box(ax, 6.5, 8.1,  3.2, 0.68, C["enc"],   "ENCProjection",
    "lon/lat → km")

# Policy core
box(ax, 4.4, 7.05, 8.0, 0.85, C["policy"],"AISReplayPolicy",
    "project · pad · normalise · pre-compute δpos & δheading", bold=True, fontsize=10)

# Stored attributes
box(ax, 1.6, 5.85, 2.8, 0.68, C["policy"],"_actions",
    "(n_steps, N, 3)  clipped [-1,1]")
box(ax, 4.5, 5.85, 2.8, 0.68, C["policy"],"initial_positions()\ngoal_positions()", fontsize=8)
box(ax, 7.5, 5.85, 2.8, 0.68, C["policy"],"initial_angles()\nt_grid  [Unix s]", fontsize=8)

# Simulation loop
box(ax, 4.5, 4.6,  8.0, 1.05, C["sim"],
    "Simulation Loop",
    "state=env.reset() → apply initial_angles → for t: actions=policy(obs,state) → obs,state=env.step()",
    bold=True, fontsize=9)

box(ax, 4.5, 3.3,  8.0, 0.72, C["sim"],
    "state_seq  [State₀ … StateN]",
    "collected every step (N × 721 frames)")

arr(ax, 2.5, 7.75, 3.4, 7.47)
arr(ax, 6.5, 7.75, 5.6, 7.47)
arr(ax, 4.4, 6.62, 1.6, 6.20)
arr(ax, 4.4, 6.62, 4.5, 6.20)
arr(ax, 4.4, 6.62, 7.5, 6.20)
arr(ax, 4.5, 5.51, 4.5, 5.13, "initial pos/goals  →  set_fixed_positions()")
arr(ax, 1.6, 5.51, 3.5, 5.13, "_actions[t]")
arr(ax, 4.5, 4.08, 4.5, 3.67)

# aligned → policy
arr(ax, 3.3, 10.29, 2.5, 8.45)
# ENCProjection → policy
arr(ax, 9.5, 12.52, 6.5, 8.45, "shared proj", conn="arc3,rad=0.1")
# env → loop
arr(ax, 12.7, 11.35, 8.5, 5.13, "obs · state · reward", conn="arc3,rad=0.05",
    lw=1.5, color="#9B59B6")
# loop → env (actions)
arr(ax, 8.5, 5.13, 13.3, 12.47, "actions", conn="arc3,rad=0.05",
    lw=1.5, color="#C0392B")

# ===========================================================================
# ⑥ RENDERING
# ===========================================================================
box(ax, 19.9, 7.95, 5.6, 0.80, C["render"],"SVGVisualizer",
    "state_seq · t_grid · fps · scale", bold=True, fontsize=10)
box(ax, 19.9, 6.85, 5.6, 0.68, C["render"],"_render_zones()",
    "anchorages & TSS polygons + direction arrows")
box(ax, 19.9, 5.85, 5.6, 0.68, C["render"],"_render_animated_objects()",
    "landmarks & goals  →  <circle> + <animate>")
box(ax, 19.9, 4.85, 5.6, 0.68, C["render"],"_render_animated_agents_with_heading()",
    "<g> animateTransform(translate) + rotate(heading)")
box(ax, 19.9, 3.85, 5.6, 0.68, C["render"],"_render_timestamp_overlay()",
    "<animate calcMode=discrete>  UTC clock from t_grid")
box(ax, 19.9, 2.8,  5.6, 0.72, C["file"],  "images/ais_replay.svg",
    "animated SVG · 721 frames · 300 vessels", bold=True)

arr(ax, 19.9, 7.55, 19.9, 7.20)
arr(ax, 19.9, 6.51, 19.9, 6.20)
arr(ax, 19.9, 5.51, 19.9, 5.20)
arr(ax, 19.9, 4.51, 19.9, 4.20)
arr(ax, 19.9, 3.51, 19.9, 3.17)

# state_seq → renderer
arr(ax, 8.5, 3.3, 17.1, 7.55, "state_seq",
    conn="arc3,rad=-0.15", lw=1.5, color="#16A085")
# t_grid → renderer
arr(ax, 7.5, 5.51, 17.1, 3.85, "t_grid",
    conn="arc3,rad=-0.1", lw=1.3, color="#16A085")
# enc_map zones → renderer
arr(ax, 12.1, 13.0, 17.1, 6.85, "zones",
    conn="arc3,rad=0.15", lw=1.2, color="#27AE60")

# ===========================================================================
# TITLE
# ===========================================================================
ax.text(13.0, 16.82,
        "CAMAR Maritime Simulator — Architecture",
        ha="center", va="center", fontsize=16, color="white", fontweight="bold")
ax.text(13.0, 16.40,
        "AIS data pipeline  ·  ENC map  ·  DeltaPosDynamic  ·  AIS replay policy  ·  SVG rendering",
        ha="center", va="center", fontsize=9, color="#AAAAAA")

# ===========================================================================
# LEGEND  (bottom-left, outside sections)
# ===========================================================================
items = [
    (C["file"],   "External file / output"),
    (C["render"], "Rendering"),
    (C["sim"],    "Simulation loop"),
    (C["policy"], "AIS replay policy"),
    (C["env"],    "CAMAR environment"),
    (C["dyn"],    "Dynamics"),
    (C["enc"],    "ENC map / projection"),
    (C["data"],   "AIS data pipeline"),
]
lx = 0.35
# stack upward from bottom
ly_start = 0.22
ax.text(lx, ly_start + len(items) * 0.35 + 0.05, "Legend",
        fontsize=8, color="#AAAAAA", fontweight="bold")
for i, (color, label) in enumerate(items):
    ly = ly_start + i * 0.35
    ax.add_patch(FancyBboxPatch((lx, ly), 0.30, 0.26,
                                boxstyle="round,pad=0.02",
                                facecolor=color, edgecolor="none", zorder=6))
    ax.text(lx + 0.42, ly + 0.13, label, va="center",
            fontsize=7.5, color="#DDDDDD", zorder=7)

plt.tight_layout(pad=0.2)
out = "images/architecture.png"
plt.savefig(out, dpi=160, bbox_inches="tight",
            facecolor=C["bg"], edgecolor="none")
print(f"Saved → {out}")
