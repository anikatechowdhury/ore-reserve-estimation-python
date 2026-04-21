# ore-reserve-estimation-python

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Plotly](https://img.shields.io/badge/Plotly-Interactive_3D-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A Python-based ore reserve estimation and 3D orebody modelling workflow applied to a chromite exploration dataset from Karnataka, India. Implements the complete chain from 2D cross section construction through reserve estimation, 3D borehole visualisation, block model grade interpolation, and open pit mine design.

---

## Overview

This project solves a real chromite orebody exploration problem computationally — replicating the workflow typically performed manually on graph paper or in commercial mining software (Surpac, Vulcan, Leapfrog). Three diamond drill holes (KB-007, KB-009, KB-011) intersecting a NS-trending chromite orebody are used to construct a scaled cross section, estimate ore reserves using the polygon method, build a 3D block model with IDW grade interpolation, and design a preliminary open pit.

---

## Dataset

| BH No | East (m) | RL Collar (m) | Dip & Direction | Depth (m) | Min. From–To (m) | Cr₂O₃ (%) |
|---|---|---|---|---|---|---|
| KB-007 | 70 | 500 | 38° → W | 80 | 25–63 | 46 |
| KB-009 | 114 | 500 | 40° → W | 140 | 83–122 | 43 |
| KB-011 | 163 | 500 | 40° → W | 200 | 146–185 | 40 |

**SG = 4.5 | Cutoff = 35% Cr₂O₃ | Scale = 1cm:10m**

---

## Modules

### Module 1 — 2D Cross Section (X–Y)
- Borehole traces plotted at correct dip angles from collar
- Mineralised intervals highlighted with grade labels
- Orebody outline connecting all intersections
- Depth tick marks at 50m intervals
- Scale bar and north arrow

### Module 2 — Ore Reserve Estimation (Polygon Method)
- Influence width per drill hole (half-distance method)
- True width from apparent width × cos(dip)
- Tonnage = Volume × SG
- Weighted average Cr₂O₃ grade
- Reserve classification table

### Module 3 — 3D Borehole Visualisation (Plotly)
- Collar positions in 3D space
- Borehole traces at correct dip/azimuth
- Mineralised intervals highlighted in orange
- Ground surface plane overlay
- Interactive HTML — rotate, zoom, pan

### Module 4 — 3D Ore Body Block Model (IDW)
- 5×5×5 m block model construction
- IDW grade interpolation from drill hole control points
- Ore blocks filtered by cutoff grade (35% Cr₂O₃)
- Blocks colour-coded by grade (YlOrRd scale)
- Orebody wireframe envelope overlay

### Module 5 — 3D Open Pit Mine Design
- Pit shell expanding with depth at 45° slope
- 10m bench height design
- Bench-level strip ratio computation
- Ore vs waste tonnage by depth
- Interactive 3D pit + ore body + drill holes

---

## Key Results

| Parameter | Value |
|---|---|
| Total Ore Tonnage | ~938,000 t |
| Average Grade | ~42.89% Cr₂O₃ |
| Total Cr₂O₃ Metal | ~402,500 t |
| Overall Strip Ratio | ~9:1 |
| Resource Classification | Indicated |

---

## Project Structure

```
ore-reserve-estimation-python/
│
├── ore_modelling.py         # Main script (all 5 modules)
├── outputs/
│   ├── cross_section_XY.png
│   ├── ore_reserve.png
│   ├── ore_reserve.csv
│   ├── borehole_3d.html        ← interactive
│   ├── ore_body_3d.html        ← interactive
│   ├── mine_design_3d.html     ← interactive
│   ├── strip_ratio.png
│   └── mine_design.csv
└── README.md
```

---

## Dependencies

| Library | Purpose |
|---|---|
| `numpy` | Geometry, coordinate transforms |
| `pandas` | Data handling, reserve table |
| `matplotlib` | 2D cross section, reserve plots |
| `plotly` | Interactive 3D visualisation |
| `scipy` | IDW interpolation (block model) |

```bash
pip install numpy pandas matplotlib plotly scipy
```

---

## Usage

```bash
python ore_modelling.py
```

Open `.html` files in any browser for interactive 3D views.

---

## Note on Extensions

This workflow can be extended using:
- **GemPy** — implicit surface modelling for orebody wireframing
- **PyVista** — advanced mesh visualisation and volumetric rendering
- **mplstereonet** — structural orientation analysis of drill hole data

---

## Author

**Anikate Chowdhury**
M.Sc. Applied Geology (2nd Year), Presidency University, Kolkata
GitHub: [github.com/anikatechowdhury](https://github.com/anikatechowdhury)

---

## License

MIT License
