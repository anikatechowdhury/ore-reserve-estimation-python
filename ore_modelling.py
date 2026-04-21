"""
Ore Reserve Estimation and 3D Orebody Modelling
-------------------------------------------------
Based on chromite orebody exploration data — Karnataka, India
(Sukinda-type chromite, NS-trending orebody)

Drill hole data:
  KB-007: East=70m, RL=500m, 38°→W, depth=80m, min=25-63m, Cr2O3=46%
  KB-009: East=114m, RL=500m, 40°→W, depth=140m, min=83-122m, Cr2O3=43%
  KB-011: East=163m, RL=500m, 40°→W, depth=200m, min=146-185m, Cr2O3=40%

Scale: 1cm = 10m | SG = 4.5 | Cutoff grade = 35% Cr2O3

Modules:
  1 — 2D Cross Section (X-Y, to scale)
  2 — Ore Reserve Estimation (polygon method + grade-tonnage)
  3 — 3D Borehole Visualisation (Plotly interactive)
  4 — 3D Ore Body Modelling (block model + IDW + wireframe)
  5 — 3D Mine Design (open pit shell + benches + strip ratio)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import os

warnings.filterwarnings("ignore")

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DRILL HOLE DATA
# ═══════════════════════════════════════════════════════════════════════════════

holes = pd.DataFrame({
    "BH":        ["KB-007", "KB-009", "KB-011"],
    "East":      [70,  114, 163],
    "RL":        [500, 500, 500],
    "North":     [0,   0,   0],       # NS trending — same northing
    "Dip":       [38,  40,  40],      # degrees from horizontal
    "Azimuth":   [270, 270, 270],     # West = 270°
    "Depth":     [80,  140, 200],
    "From":      [25,  83,  146],
    "To":        [63,  122, 185],
    "Grade":     [46,  43,  40],
})

holes["IntLen"] = holes["To"] - holes["From"]
holes["MidDepth"] = (holes["From"] + holes["To"]) / 2
SG      = 4.5
CUTOFF  = 35.0
SCALE   = 10    # 1cm = 10m

print("── Drill Hole Data ──")
print(holes[["BH","East","RL","Dip","From","To","IntLen","Grade"]].to_string(index=False))

# ── Compute 3D coordinates of each point along drill hole ────────────────────
def borehole_coords(east, rl, north, dip_deg, az_deg, depths):
    """Convert depth along hole to 3D XYZ coordinates."""
    dip_rad = np.radians(dip_deg)
    az_rad  = np.radians(az_deg)
    xs, ys, zs = [], [], []
    for d in depths:
        # horizontal distance
        h = d * np.cos(dip_rad)
        # vertical drop
        v = d * np.sin(dip_rad)
        dx = h * np.sin(az_rad)   # East component
        dy = h * np.cos(az_rad)   # North component
        xs.append(east + dx)
        ys.append(north + dy)
        zs.append(rl - v)
    return np.array(xs), np.array(ys), np.array(zs)

# Compute key points for each hole
for i, row in holes.iterrows():
    d_arr = np.array([0, row["From"], row["To"], row["Depth"]])
    xs, ys, zs = borehole_coords(
        row["East"], row["RL"], row["North"],
        row["Dip"], row["Azimuth"], d_arr)
    holes.loc[i, "X_collar"] = xs[0]
    holes.loc[i, "Y_collar"] = ys[0]
    holes.loc[i, "Z_collar"] = zs[0]
    holes.loc[i, "X_from"]   = xs[1]
    holes.loc[i, "Z_from"]   = zs[1]
    holes.loc[i, "X_to"]     = xs[2]
    holes.loc[i, "Z_to"]     = zs[2]
    holes.loc[i, "X_end"]    = xs[3]
    holes.loc[i, "Z_end"]    = zs[3]
    holes.loc[i, "X_mid"]    = (xs[1] + xs[2]) / 2
    holes.loc[i, "Z_mid"]    = (zs[1] + zs[2]) / 2

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — 2D CROSS SECTION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n── Module 1: 2D Cross Section ──")

fig, ax = plt.subplots(figsize=(14, 10))

# Surface line
x_surf = np.linspace(0, 250, 300)
z_surf = np.full(300, 500.0)
ax.plot(x_surf, z_surf, "k-", linewidth=2.5, zorder=5)
ax.fill_between(x_surf, z_surf, 300, color="#D7CCC8", alpha=0.5)

# Grid
for z in range(300, 510, 50):
    ax.axhline(z, color="lightgrey", linewidth=0.5, linestyle="--", alpha=0.6)
for x in range(0, 260, 50):
    ax.axvline(x, color="lightgrey", linewidth=0.5, linestyle="--", alpha=0.6)

bh_colors = ["#1565C0", "#2E7D32", "#B71C1C"]

for i, (_, row) in enumerate(holes.iterrows()):
    col = bh_colors[i]

    # Full borehole trace
    ax.annotate("", xy=(row["X_end"], row["Z_end"]),
                xytext=(row["X_collar"], row["Z_collar"]),
                arrowprops=dict(arrowstyle="-", color=col,
                                lw=1.8))

    # Mineralised interval — thick coloured line
    ax.plot([row["X_from"], row["X_to"]],
            [row["Z_from"], row["Z_to"]],
            color=col, linewidth=8, alpha=0.7,
            solid_capstyle="round", zorder=6)

    # Interval label
    ax.text(row["X_mid"] - 12, row["Z_mid"],
            f"{row['Grade']:.0f}% Cr₂O₃\n{row['From']:.0f}–{row['To']:.0f}m",
            fontsize=8, color=col, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", alpha=0.8,
                      edgecolor=col))

    # Collar marker + label
    ax.plot(row["X_collar"], row["Z_collar"],
            "v", color=col, markersize=12, zorder=7)
    ax.text(row["X_collar"], row["Z_collar"] + 8,
            row["BH"], fontsize=9, ha="center",
            fontweight="bold", color=col)

    # Depth tick marks every 50m
    for d in np.arange(50, row["Depth"]+1, 50):
        xs, _, zs = borehole_coords(
            row["East"], row["RL"], row["North"],
            row["Dip"], row["Azimuth"], [d])
        ax.plot(xs[0], zs[0], "+", color=col,
                markersize=6, zorder=4)
        ax.text(xs[0] + 3, zs[0], f"{d:.0f}",
                fontsize=6, color=col, alpha=0.8)

# Orebody outline — connect mineralised intersections
ore_x = list(holes["X_from"]) + list(holes["X_to"])[::-1]
ore_z = list(holes["Z_from"]) + list(holes["Z_to"])[::-1]
ore_x.append(ore_x[0]); ore_z.append(ore_z[0])
ax.fill(ore_x, ore_z, color="#FF6F00", alpha=0.25,
        zorder=3, label="Orebody outline")
ax.plot(ore_x, ore_z, "k--", linewidth=1.5,
        zorder=4, alpha=0.7)

# Scale bar
ax.plot([10, 110], [310, 310], "k-", linewidth=3)
ax.text(60, 305, "100 m", ha="center", fontsize=9,
        fontweight="bold")

# Compass
ax.annotate("N↑", xy=(230, 490), fontsize=12,
            fontweight="bold", ha="center")

# Legend
patches = [mpatches.Patch(color=bh_colors[i],
                           label=holes["BH"].iloc[i])
           for i in range(3)]
patches.append(mpatches.Patch(color="#FF6F00",
                               alpha=0.5,
                               label="Mineralised zone"))
ax.legend(handles=patches, loc="lower right", fontsize=9)

ax.set_xlim(0, 250)
ax.set_ylim(295, 520)
ax.set_xlabel("Easting (m)", fontsize=11)
ax.set_ylabel("RL / Elevation (m)", fontsize=11)
ax.set_title("Cross Section X–Y — Chromite Orebody\n"
             "NS-Trending Orebody | Scale 1:1000 (1cm = 10m)",
             fontsize=13, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/cross_section_XY.png",
            dpi=180, bbox_inches="tight")
plt.close()
print("Saved: cross_section_XY.png")

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — ORE RESERVE ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n── Module 2: Ore Reserve Estimation ──")

# Polygon of influence method
# Each drill hole influences half the distance to adjacent holes
# Area of influence × true width × SG = tonnage

# Distances between holes (East direction)
e = holes["East"].values
d_07_09 = (e[1] - e[0]) / 2   # 22m
d_09_11 = (e[2] - e[1]) / 2   # 24.5m

# Boundary extensions (assume half-spacing beyond outermost holes)
ext_left  = d_07_09    # 22m
ext_right = d_09_11    # 24.5m

influence_widths = [
    ext_left + d_07_09,          # KB-007: 44m
    d_07_09 + d_09_11,           # KB-009: 46.5m
    d_09_11 + ext_right,         # KB-011: 49m
]

# True width = interval length × cos(dip)
true_widths = holes["IntLen"] * np.cos(np.radians(holes["Dip"]))

# Assume NS extent = 50m (50m interval drilling stated)
NS_EXTENT = 50.0

# Volume and tonnage per hole
reserves = []
for i, (_, row) in enumerate(holes.iterrows()):
    vol  = influence_widths[i] * true_widths.iloc[i] * NS_EXTENT
    tons = vol * SG
    reserves.append({
        "BH":              row["BH"],
        "Influence_EW_m":  round(influence_widths[i], 1),
        "True_Width_m":    round(true_widths.iloc[i], 1),
        "NS_Extent_m":     NS_EXTENT,
        "Volume_m3":       round(vol, 1),
        "Tonnage_t":       round(tons, 1),
        "Grade_Cr2O3_pct": row["Grade"],
        "Metal_t":         round(tons * row["Grade"] / 100, 1),
    })

res_df = pd.DataFrame(reserves)

# Total / weighted average
total_tons  = res_df["Tonnage_t"].sum()
total_metal = res_df["Metal_t"].sum()
avg_grade   = total_metal / total_tons * 100

res_df.loc[len(res_df)] = {
    "BH": "TOTAL",
    "Influence_EW_m": "",
    "True_Width_m": "",
    "NS_Extent_m": "",
    "Volume_m3": res_df["Volume_m3"].sum(),
    "Tonnage_t": round(total_tons, 1),
    "Grade_Cr2O3_pct": round(avg_grade, 2),
    "Metal_t": round(total_metal, 1),
}

res_df.to_csv(f"{OUTPUT_DIR}/ore_reserve.csv", index=False)
print(res_df.to_string(index=False))
print(f"\n  Total Tonnage : {total_tons:,.0f} t")
print(f"  Average Grade : {avg_grade:.2f}% Cr₂O₃")
print(f"  Total Metal   : {total_metal:,.0f} t Cr₂O₃")

# ── Plot reserve summary ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Tonnage bar
ax = axes[0]
bhs = holes["BH"].tolist()
tons = res_df[res_df["BH"] != "TOTAL"]["Tonnage_t"].values
ax.bar(bhs, tons, color=bh_colors, edgecolor="white", width=0.5)
for b, t in zip(bhs, tons):
    ax.text(bhs.index(b), t + 500, f"{t:,.0f}t",
            ha="center", fontsize=9, fontweight="bold")
ax.set_ylabel("Tonnage (t)", fontsize=10)
ax.set_title("Tonnage by Drill Hole", fontsize=11, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Grade bar
ax = axes[1]
grades = holes["Grade"].values
bars = ax.bar(bhs, grades, color=bh_colors, edgecolor="white", width=0.5)
ax.axhline(CUTOFF, color="red", linestyle="--", linewidth=1.5,
           label=f"Cutoff = {CUTOFF}%")
ax.axhline(avg_grade, color="navy", linestyle="-.", linewidth=1.5,
           label=f"Avg = {avg_grade:.1f}%")
for b, g in zip(bhs, grades):
    ax.text(bhs.index(b), g + 0.5, f"{g}%",
            ha="center", fontsize=9, fontweight="bold")
ax.set_ylabel("Cr₂O₃ Grade (%)", fontsize=10)
ax.set_title("Grade by Drill Hole", fontsize=11, fontweight="bold")
ax.legend(fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Grade-Tonnage summary
ax = axes[2]
ax.axis("off")
table_data = [
    ["Parameter", "Value"],
    ["Total Tonnage", f"{total_tons:,.0f} t"],
    ["Average Grade", f"{avg_grade:.2f}% Cr₂O₃"],
    ["Total Metal", f"{total_metal:,.0f} t"],
    ["SG", f"{SG}"],
    ["NS Extent", f"{NS_EXTENT} m"],
    ["Cutoff Grade", f"{CUTOFF}% Cr₂O₃"],
    ["Classification", "Indicated"],
]
tbl = ax.table(cellText=table_data,
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.3, 1.8)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#1565C0")
        cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#E3F2FD")
ax.set_title("Reserve Summary", fontsize=11, fontweight="bold")

fig.suptitle("Ore Reserve Estimation — Polygon Method\n"
             "Chromite Orebody, Karnataka",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/ore_reserve.png",
            dpi=180, bbox_inches="tight")
plt.close()
print("\nSaved: ore_reserve.png")

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — 3D BOREHOLE VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n── Module 3: 3D Borehole Visualisation ──")

fig3d = go.Figure()

plotly_colors = ["#1565C0", "#2E7D32", "#B71C1C"]

for i, (_, row) in enumerate(holes.iterrows()):
    col = plotly_colors[i]

    # Full hole trace
    n_pts = 50
    depths_full = np.linspace(0, row["Depth"], n_pts)
    xs, ys, zs = borehole_coords(
        row["East"], row["RL"], row["North"],
        row["Dip"], row["Azimuth"], depths_full)

    fig3d.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line=dict(color=col, width=4),
        name=f"{row['BH']} trace",
        showlegend=True,
    ))

    # Mineralised interval
    depths_min = np.linspace(row["From"], row["To"], 20)
    xm, ym, zm = borehole_coords(
        row["East"], row["RL"], row["North"],
        row["Dip"], row["Azimuth"], depths_min)

    fig3d.add_trace(go.Scatter3d(
        x=xm, y=ym, z=zm,
        mode="lines",
        line=dict(color="#FF6F00", width=10),
        name=f"{row['BH']} min ({row['Grade']}% Cr₂O₃)",
        showlegend=True,
    ))

    # Collar marker
    fig3d.add_trace(go.Scatter3d(
        x=[row["X_collar"]], y=[row["Y_collar"]],
        z=[row["Z_collar"]],
        mode="markers+text",
        marker=dict(size=8, color=col, symbol="diamond"),
        text=[row["BH"]],
        textposition="top center",
        name=f"{row['BH']} collar",
        showlegend=False,
    ))

# Surface plane
x_s = np.linspace(40, 200, 10)
y_s = np.linspace(-30, 30, 10)
XX, YY = np.meshgrid(x_s, y_s)
ZZ = np.full_like(XX, 500.0)

fig3d.add_trace(go.Surface(
    x=XX, y=YY, z=ZZ,
    colorscale=[[0, "rgba(139,115,85,0.3)"],
                [1, "rgba(139,115,85,0.3)"]],
    showscale=False, name="Ground surface",
    opacity=0.3,
))

fig3d.update_layout(
    title=dict(text="3D Borehole Visualisation — Chromite Orebody<br>"
               "<sub>Orange = Mineralised intervals | Karnataka, India</sub>",
               font=dict(size=14)),
    scene=dict(
        xaxis_title="Easting (m)",
        yaxis_title="Northing (m)",
        zaxis_title="RL (m)",
        camera=dict(eye=dict(x=1.8, y=1.8, z=0.8)),
        bgcolor="rgb(245,245,250)",
    ),
    width=900, height=650,
    showlegend=True,
)

fig3d.write_html(f"{OUTPUT_DIR}/borehole_3d.html")

# Static screenshot
print("Saved: borehole_3d.html")

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — 3D ORE BODY MODELLING
# ═══════════════════════════════════════════════════════════════════════════════

print("\n── Module 4: 3D Ore Body Modelling ──")

# Build 3D block model — 5m × 5m × 5m blocks
BX, BY, BZ = 5, 5, 5   # block sizes (m)

x_range = np.arange(40, 200, BX)
y_range = np.arange(-30, 30, BY)
z_range = np.arange(300, 505, BZ)

# Grade control points from drill holes (midpoint of mineralised interval)
ctrl_pts = np.array([
    [holes["X_mid"].iloc[i], 0, holes["Z_mid"].iloc[i]]
    for i in range(len(holes))
])
ctrl_grades = holes["Grade"].values.astype(float)

# IDW interpolation for block model
def idw_interp(query_pts, ctrl_pts, values, power=2):
    dists = cdist(query_pts, ctrl_pts)
    dists = np.where(dists == 0, 1e-10, dists)
    weights = 1.0 / dists**power
    return np.sum(weights * values, axis=1) / np.sum(weights, axis=1)

# Build block centres
blocks = []
for bx in x_range:
    for by in y_range:
        for bz in z_range:
            blocks.append([bx + BX/2, by + BY/2, bz + BZ/2])

blocks = np.array(blocks)
grades_idw = idw_interp(blocks, ctrl_pts, ctrl_grades)

# Filter — only show blocks near the orebody
# Use distance to nearest control point
dists_to_ore = cdist(blocks, ctrl_pts).min(axis=1)
ore_mask = (grades_idw >= CUTOFF) & (dists_to_ore < 60)

ore_blocks = blocks[ore_mask]
ore_grades = grades_idw[ore_mask]

print(f"  Total blocks: {len(blocks)}")
print(f"  Ore blocks (>{CUTOFF}% Cr₂O₃): {ore_mask.sum()}")
print(f"  Block model grade range: {ore_grades.min():.1f}–{ore_grades.max():.1f}%")

# 3D block model plot
fig4 = go.Figure()

# Ore blocks coloured by grade
fig4.add_trace(go.Scatter3d(
    x=ore_blocks[:, 0],
    y=ore_blocks[:, 1],
    z=ore_blocks[:, 2],
    mode="markers",
    marker=dict(
        size=4,
        color=ore_grades,
        colorscale="YlOrRd",
        colorbar=dict(title="Cr₂O₃ (%)",
                      x=1.05),
        opacity=0.75,
        cmin=CUTOFF,
        cmax=50,
    ),
    name="Ore blocks",
    text=[f"Grade: {g:.1f}%" for g in ore_grades],
    hovertemplate="E: %{x}m<br>N: %{y}m<br>RL: %{z}m<br>%{text}",
))

# Drill hole traces overlay
for i, (_, row) in enumerate(holes.iterrows()):
    depths_full = np.linspace(0, row["Depth"], 40)
    xs, ys, zs = borehole_coords(
        row["East"], row["RL"], row["North"],
        row["Dip"], row["Azimuth"], depths_full)
    fig4.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line=dict(color=plotly_colors[i], width=4),
        name=row["BH"],
        showlegend=True,
    ))

# Orebody wireframe envelope
# Connect mineralised interval endpoints
wire_x = ([holes["X_from"].iloc[j] for j in range(3)] +
          [holes["X_to"].iloc[j] for j in range(3)] +
          [holes["X_from"].iloc[0]])
wire_y = [0] * 7
wire_z = ([holes["Z_from"].iloc[j] for j in range(3)] +
          [holes["Z_to"].iloc[j] for j in range(3)] +
          [holes["Z_from"].iloc[0]])

fig4.add_trace(go.Scatter3d(
    x=wire_x, y=wire_z, z=wire_y,
    mode="lines",
    line=dict(color="black", width=3, dash="dash"),
    name="Orebody wireframe",
))

fig4.update_layout(
    title=dict(
        text="3D Ore Body Block Model — IDW Grade Interpolation<br>"
             "<sub>Chromite Orebody | Blocks coloured by Cr₂O₃ grade</sub>",
        font=dict(size=14)),
    scene=dict(
        xaxis_title="Easting (m)",
        yaxis_title="Northing (m)",
        zaxis_title="RL (m)",
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        bgcolor="rgb(240,240,248)",
    ),
    width=950, height=700,
)

fig4.write_html(f"{OUTPUT_DIR}/ore_body_3d.html")
print("Saved: ore_body_3d.html")

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 5 — 3D MINE DESIGN
# ═══════════════════════════════════════════════════════════════════════════════

print("\n── Module 5: 3D Mine Design ──")

# Open pit design parameters
BENCH_HEIGHT  = 10    # m
SLOPE_ANGLE   = 45    # degrees
CREST_WIDTH   = 5     # m (safety berm)
PIT_BOTTOM_RL = 350   # m RL

# Generate pit shell — expanding with depth
pit_levels = np.arange(500, PIT_BOTTOM_RL - 1, -BENCH_HEIGHT)

# Orebody centre
ore_cx = np.mean(holes["X_mid"].values)
ore_cz = np.mean(holes["Z_mid"].values)

pit_traces = []
for rl in pit_levels:
    depth = 500 - rl
    expansion = depth / np.tan(np.radians(SLOPE_ANGLE))
    x_min = ore_cx - 30 - expansion
    x_max = ore_cx + 30 + expansion
    y_min = -20 - expansion * 0.3
    y_max =  20 + expansion * 0.3
    pit_traces.append({
        "RL": rl, "depth": depth,
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
    })

pit_df = pd.DataFrame(pit_traces)

# Strip ratio computation
# Ore tonnes at each level vs waste
ore_per_level   = []
waste_per_level = []
for _, pt in pit_df.iterrows():
    area_total = (pt["x_max"] - pt["x_min"]) * (pt["y_max"] - pt["y_min"])
    # Simple ore fraction — ore body width / total pit width
    ore_width  = max(0, min(60.0, pt["x_max"] - pt["x_min"]) * 0.25)
    ore_area   = ore_width * (pt["y_max"] - pt["y_min"])
    waste_area = area_total - ore_area
    ore_per_level.append(ore_area * BENCH_HEIGHT * SG)
    waste_per_level.append(waste_area * BENCH_HEIGHT * 2.7)

pit_df["Ore_t"]   = ore_per_level
pit_df["Waste_t"] = waste_per_level
pit_df["SR"]      = pit_df["Waste_t"] / pit_df["Ore_t"]

total_ore   = pit_df["Ore_t"].sum()
total_waste = pit_df["Waste_t"].sum()
overall_sr  = total_waste / total_ore
print(f"  Overall Strip Ratio: {overall_sr:.2f}:1")
print(f"  Total Ore  : {total_ore:,.0f} t")
print(f"  Total Waste: {total_waste:,.0f} t")

pit_df.to_csv(f"{OUTPUT_DIR}/mine_design.csv", index=False)

# 3D pit shell
fig5 = go.Figure()

# Draw bench polygons
for idx, pt in pit_df.iterrows():
    alpha = max(0.05, 0.4 - idx * 0.025)
    fig5.add_trace(go.Scatter3d(
        x=[pt["x_min"], pt["x_max"], pt["x_max"],
           pt["x_min"], pt["x_min"]],
        y=[pt["y_min"], pt["y_min"], pt["y_max"],
           pt["y_max"], pt["y_min"]],
        z=[pt["RL"]] * 5,
        mode="lines",
        line=dict(color=f"rgba(150,75,0,0.6)", width=1.5),
        showlegend=False,
    ))

# Ore body blocks (from module 4)
fig5.add_trace(go.Scatter3d(
    x=ore_blocks[:, 0],
    y=ore_blocks[:, 1],
    z=ore_blocks[:, 2],
    mode="markers",
    marker=dict(
        size=3,
        color=ore_grades,
        colorscale="YlOrRd",
        opacity=0.8,
        cmin=CUTOFF, cmax=50,
        colorbar=dict(title="Cr₂O₃ (%)"),
    ),
    name="Ore blocks",
))

# Drill holes
for i, (_, row) in enumerate(holes.iterrows()):
    depths_full = np.linspace(0, row["Depth"], 40)
    xs, ys, zs = borehole_coords(
        row["East"], row["RL"], row["North"],
        row["Dip"], row["Azimuth"], depths_full)
    fig5.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line=dict(color=plotly_colors[i], width=3),
        name=row["BH"],
    ))

# Pit bottom
pb = pit_df.iloc[-1]
fig5.add_trace(go.Scatter3d(
    x=[pb["x_min"], pb["x_max"], pb["x_max"],
       pb["x_min"], pb["x_min"]],
    y=[pb["y_min"], pb["y_min"], pb["y_max"],
       pb["y_max"], pb["y_min"]],
    z=[PIT_BOTTOM_RL] * 5,
    mode="lines",
    line=dict(color="red", width=3),
    name=f"Pit bottom (RL {PIT_BOTTOM_RL}m)",
))

fig5.update_layout(
    title=dict(
        text="3D Open Pit Mine Design<br>"
             f"<sub>Bench height={BENCH_HEIGHT}m | "
             f"Slope={SLOPE_ANGLE}° | "
             f"Strip ratio={overall_sr:.2f}:1</sub>",
        font=dict(size=14)),
    scene=dict(
        xaxis_title="Easting (m)",
        yaxis_title="Northing (m)",
        zaxis_title="RL (m)",
        camera=dict(eye=dict(x=1.8, y=1.8, z=1.2)),
        bgcolor="rgb(235,235,245)",
    ),
    width=950, height=700,
)

fig5.write_html(f"{OUTPUT_DIR}/mine_design_3d.html")
print("Saved: mine_design_3d.html")

# ── Strip ratio plot ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(pit_df["SR"], pit_df["RL"], "o-",
        color="#B71C1C", linewidth=2, markersize=5)
ax.axvline(overall_sr, color="navy", linestyle="--",
           linewidth=1.5,
           label=f"Overall SR = {overall_sr:.2f}:1")
ax.set_xlabel("Strip Ratio (waste:ore)", fontsize=10)
ax.set_ylabel("RL (m)", fontsize=10)
ax.set_title("Strip Ratio by Bench Level",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax = axes[1]
depths_plot = 500 - pit_df["RL"]
ax.fill_betweenx(depths_plot, 0,
                  pit_df["Ore_t"]/1e3,
                  color="#FF6F00", alpha=0.7, label="Ore (kt)")
ax.fill_betweenx(depths_plot,
                  pit_df["Ore_t"]/1e3,
                  (pit_df["Ore_t"] + pit_df["Waste_t"])/1e3,
                  color="#78909C", alpha=0.6, label="Waste (kt)")
ax.set_xlabel("Tonnage (kt per bench)", fontsize=10)
ax.set_ylabel("Depth below surface (m)", fontsize=10)
ax.set_title("Ore vs Waste Tonnage by Depth",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.invert_yaxis()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.suptitle("Open Pit Mine Design — Strip Ratio Analysis",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/strip_ratio.png",
            dpi=180, bbox_inches="tight")
plt.close()
print("Saved: strip_ratio.png")

print(f"\n✓ All outputs saved to: {OUTPUT_DIR}")
print(f"  Module 1 — Cross Section : cross_section_XY.png")
print(f"  Module 2 — Reserve       : ore_reserve.png + ore_reserve.csv")
print(f"  Module 3 — 3D Boreholes  : borehole_3d.html (interactive)")
print(f"  Module 4 — 3D Ore Body   : ore_body_3d.html (interactive)")
print(f"  Module 5 — Mine Design   : mine_design_3d.html (interactive)")
print(f"             Strip Ratio   : strip_ratio.png + mine_design.csv")
