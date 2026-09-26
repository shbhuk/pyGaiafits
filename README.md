# pyGaiafits

Publicly released alongside the manuscript "A Joint Astrometric and Radial Velocity Study of Gaia-4b" by Kanodia et al. (in prep.)

# Overview

`xo_Astrometry.py` fits Gaia DR4 pre-release epoch astrometry — the per-CCD
along-scan (AL) abscissae — for a single source, using PyMC3 and `exoplanet`.
It always fits a 5-parameter single-star model first, then optionally adds a
Keplerian photocentre orbit and a jointly-fit radial velocity model, seeded
from the single-star MAP solution. Each stage is MAP-optimized and then
sampled with NUTS.

Only the along-scan direction enters the likelihood; the across-scan
measurements are not used. These are typically only relevant for bright stars for Gaia.


# Setup

## Environment

Follow the steps below to create a [virtual environment](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/), and then activate it.

```
module load anaconda3
conda create -p=~/work/[INSERT VENV NAME HERE] python=3.9
conda activate ~/work/[INSERT VENV NAME HERE]
which python # This should point to the path above
```
Install the packages with versions as shown below - 

## Requirements (last tested on)
```
numpy==1.20.3
scipy==1.8.1
pandas==1.1.4 # Suggestion
matplotlib==3.2.2
arviz==0.11.1
theano==1.0.5
pymc3==3.9.3
exoplanet==0.4.4
uncertainties==3.1.5
astropy==4.2

pymc3-ext==0.0.2
corner==2.2.1
celerite2==0.1.0

```

Then you will follow the instructions below and clone this GitHub repository. See tips [here](https://www.narenvadapalli.com/blog/github-login-using-access-token-via-cmdline/) and [here](https://stackoverflow.com/questions/2505096/clone-a-private-repository-github).

## Directory Structure
You need to have a `DataParentDirectory` with a separate directory for each star (e.g. `Gaia4`, etc.), and then point to this path in 
`Code\Config.py`. The scripts will refer to `DataParentDirectory` for the astrometry dataset for DR4 pre-release, but can be changed later for DR4. 
I suggest using this same directory for the RV data files too, but technically the RV file can be stored anywhere.

```
pyGaiafits
|   Code


DataParentDirectory # Point to this path in Code/Config.py
    GAIADR4PreReleaseXML.XML
|   Star1Name   
|   |   RV_timeseries.csv
|   Star2Name
....
```

# Usage

Run from the repository root (or anywhere — the script puts its own directory
on `sys.path`):

```
# 5-parameter single-star fit
python Code/xo_Astrometry.py --StarName Gaia4 --SOURCE_ID 1457486023639239296 --Analysis Single

# single star + astrometric orbit
python Code/xo_Astrometry.py --StarName Gaia4 --SOURCE_ID 1457486023639239296 --Analysis Binary

# the above, fit jointly with radial velocities
python Code/xo_Astrometry.py --StarName Gaia4 --SOURCE_ID 1457486023639239296 \
    --Analysis "Binary+RV" --rv-csv "../Data/Gaia4/Gaia4b_GummiHARPSN_HPF2026.csv"
```

| Argument | Required | Description |
| --- | --- | --- |
| `--SOURCE_ID` | no | Gaia `source_id` to fit. If it is not in the input VOTable the script exits and lists the `source_id`s that are. |
| `--StarName` | no | Label for the output directory and filenames. |
| `--Analysis` | no | One of `Single`, `Binary`, `Binary+RV`. Defaults to `Binary`. |
| `--rv-csv` | for `Binary+RV` | Path to the combined RV file. |

Anything left out falls back to the defaults set near the top of the script,
which are currently Gaia-4 (`Gaia4`, `1457486023639239296`).

## Analysis modes

- **`Single`** — position offsets (`dRA`, `dDec`), proper motions (`PMRA`,
  `PMDec`), parallax, and an excess along-scan jitter term.
- **`Binary`** — the above plus a photocentre orbit in Campbell elements
  (P, a0, e, omega, Omega, i, T_peri), with the Thiele-Innes constants
  (A, B, F, G) recorded as derived quantities. The orbit is seeded by default
  from the Thiele-Innes grid search in `xo_utils.scan_orbit_init`.
- **`Binary+RV`** — the above plus an RV model sharing (P, T_peri, e, omega)
  with the astrometric orbit. The RV semi-amplitude K is a free parameter
  rather than derived from a0, and a per-instrument offset and jitter and a
  shared quadratic systemic trend are fit alongside.

# Input data

`DataParentDirectory` in `Code/Config.py` can point anywhere. Two things are
read from it:

```
DataParentDirectory/
├── GAIA_DR4_PRERELEASE_EPOCH_ASTROMETRY_RAW.xml   # epoch astrometry, all sources in one file. 
└── <StarName>/                                    # must exist; outputs are written here
```

The epoch-astrometry file is a single VOTable (BINARY2) holding every source;
one row per field-of-view transit, with per-CCD quantities stored as
variable-length arrays. It is read with `astropy` when available and with a
dependency-free BINARY2 reader otherwise. 
For Gaia DR4 (post December 2026) the data file and reader function should be updated.

`<StarName>/` is **not** created automatically — only the run subdirectory
inside it is, so create the per-star directory before the first run for a new
target.

## RV file format

One csv holding every instrument, one row per measurement:

| column | meaning |
| --- | --- |
| `bjd` | time, BJD |
| `rv` | radial velocity, m/s |
| `e_rv` | RV uncertainty, m/s |
| `inst` | instrument label (lower case column name) |
| `Use` | optional; rows whose value is falsy are dropped |

Per-instrument offsets and jitters are fit for each distinct `inst` value.

## Example data

`Data/` in this repository is a working example, not a required location: the
VOTable holds 12 sources, and `Data/Gaia4/Gaia4b_GummiHARPSN_HPF2026.csv`
holds 44 RVs of Gaia-4 from four instruments (HARPS-N, HPF, NEID, FIES).
Point `DataParentDirectory` at it to reproduce the published fit.

# Outputs

Written to `<DataParentDirectory>/<StarName>/<RunName>/`:

| File | Contents |
| --- | --- |
| `Plots_<RunName>.pdf` | orbit initialization, MAP and MCMC along-scan and sky plots, posterior summary table, trace plot, and the RV figures when RVs are fit |
| `ChainSummary_<RunName>.csv` | posterior summary table |
| `MCMC_Samples.csv` | posterior samples of the summarized parameters |
| `TraceSummary.png`, `TraceFigure.png` | summary table and trace plot as images |
| `CornerPlot.png` | corner plot |

Re-running the same mode on the same star reuses the directory and overwrites
these files.

# Settings not exposed on the command line

Edit these near the top of `xo_Astrometry.py`:

| Name | Default | Effect |
| --- | --- | --- |
| `GuessOrbit` | `True` | Seed the orbit from the grid search rather than from broad uninformative priors. |
| `LinRVTrend` | `False` | `True` fits a linear systemic RV trend; `False` fits linear + quadratic. |
| `Nchains` | `3` | NUTS chains, and the number of cores requested. |
| `RunName` | `'AllRVs_QuadTrend_UseEverything'` | Descriptive middle of the output directory name; it is a fixed string, so update it by hand if the settings above change. |

Sampling uses `tune=3000`, `draws=1500` per chain and dominates the runtime.


# Citation

Cite "A Joint Astrometric and Radial Velocity Study of Gaia-4b" by Kanodia et al. (in prep.)