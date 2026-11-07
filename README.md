# KeynoteTalk — XI SBGEA 2026

This standalone project contains the scripts, notebooks, scientific models,
and visualization code used to prepare the XI SBGEA 2026 keynote. The original
working copy remains in the `prizm` project. Large or personal inputs and
generated presentation assets are kept outside this repository in the
`KEYNOTE` data directory.

## Install and run from this checkout

Python 3.10 or later is required. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate                 # macOS / Linux
python -m pip install -e '.[notebooks]'
```

Install FFmpeg with H.264 support for PowerPoint video output:

```bash
brew install ffmpeg                       # macOS
sudo apt install ffmpeg                   # Debian / Ubuntu
```

Render outputs are written under `KEYNOTE/outputs/`:

```bash
python scripts/render_mission.py --master
python scripts/render_segments.py
python scripts/render_figures.py
python scripts/render_mission.py --short
python scripts/render_slides.py
```

The package also exposes `sbgea-render-mission`, `sbgea-render-projectile`,
and `sbgea-render-slides` after editable installation. If FFmpeg is missing,
the renderers that support it save PNG frames and figures; they cannot create
an H.264 MP4 without an FFmpeg encoder.

## Data location

The default data root is:

```text
~/Documents/Science/Programs/LFS/KEYNOTE/
├── inputs/
└── outputs/
```

Set `SBGEA_KEYNOTE_DATA` to point to a local data directory on another
computer. For example:

```bash
export SBGEA_KEYNOTE_DATA="$HOME/ScienceData/KEYNOTE"
```

`sbgea_keynote.paths` is the single path interface used by the Python package
and project notebooks. Input data and generated outputs are intentionally not
included in this repository. Some presentation notebooks require local source
data in `KEYNOTE/inputs`; without those files, their data-loading and rendering
cells cannot run. The three OSSE experiment notebooks and projectile
simulations are self-contained apart from their Python packages. Outputs
created by scripts go under `KEYNOTE/outputs`.

## Source layout

```text
src/sbgea_keynote/
├── config.py                 # configurable physical and rendering parameters
├── paths.py                  # centralized KEYNOTE input/output paths
├── physics/projectile.py     # projectile, drag, wind, and orbital dynamics
├── mission/                  # rocket, sphere, radar, retrieval, workflow
├── instruments/              # independent instrument forward models
└── visualization/            # visualizations and render entry points
scripts/                      # convenient commands from the source checkout
tests/                        # physics and numerical verification
notebooks/                    # project notebooks, grouped by scientific topic
```

Numerical dynamics and retrieval functions are kept separate from rendering.
The `src` layout and `pyproject.toml` support editable installation as a
standalone project.

## Notebooks and retained deliverables

The notebooks are grouped in `notebooks/mission/`, `notebooks/network/`,
`notebooks/figures/`, `notebooks/digitization/`, `notebooks/experiments/`, and `notebooks/supermag/`.
Their code uses package imports or is self-contained, as appropriate. Generated
keynote figures, animations, network plots, OSSE payloads, and digitized data
are stored in the external `KEYNOTE/outputs/` tree and are not committed here.

The Fazenda Pamplona notebook is a review/replot workflow over digitized CSVs.
The three OSSE notebooks are reconstructed, self-contained educational
simulations based on the project specification:
`notebooks/experiments/Experiment_1_FabryPerot_Cariri.ipynb`,
`notebooks/experiments/Experiment_2_MeteorRadar_CampinaGrande.ipynb`, and
`notebooks/experiments/Experiment_3_GNSS_EquatorialSpreadF.ipynb`. Their
See `notebooks/experiments/README.md` for execution, assumptions, and
provenance details. The Fabry–Pérot teaching notebook and budget infographic
notebook were recovered from local executed-kernel history.

The SuperMAG comparison notebook,
`notebooks/supermag/SuperMAG_Global_vs_EMBRACE_Brazil_2025.ipynb`, combines a
single-day global SuperMAG vector product with the 2025 EMBRACE XDR station
files. It writes coverage tables to `KEYNOTE/outputs/data/supermag/`; the broad
comparison plot stays inline in the notebook, while a Cartopy network map is
saved to `KEYNOTE/outputs/figures/supermag/SuperMAG_station_network_2025.png`. The staged SuperMAG file is only
for 10 June 2025; after normalizing its 0–360° longitudes, it contains seven
stations in the South America region box and shares VSS with the EMBRACE
files. The paired plot is a cross-product consistency check, not independent
measurements. The notebook checks the public HAPI registry; the KNMI SuperMAG mirror ends in 2023 and has no SMR. For 2025 it uses SuperMAG’s public 60-second index service, with raw minute data kept in memory.
Install the `notebooks` optional dependencies to run it (`netCDF4`, Cartopy, and Requests are included).

## Scientific scope

The keynote model begins with a 25 m/s, 45° local projectile, adds drag and
wind, then shifts to a generic sounding-rocket payload and a falling sphere.
The sphere's measured range/angle track is fit with a forward model to infer a
height-dependent neutral wind. Assumptions, uncertainties, and regime
diagnostics are documented in the code and the surviving mission notebook.
This is an educational observing-system simulation, not a flight-qualified
trajectory or instrument model and not a simulation of a historical rocket.

## Tests

The automated checks are in `tests/` and can be run after installing the
development extra:

```bash
python -m pip install -e '.[dev]'
pytest
```

They cover analytic projectile checks, numerical convergence, energy behavior,
atmospheric sensitivity, release states, synthetic observation reproducibility,
and retrieval behavior under nominal and degraded tracking. Passing these
checks is numerical verification under the implemented assumptions, not
validation against flight or instrument data.
