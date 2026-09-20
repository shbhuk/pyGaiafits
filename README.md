# pyGaiafits

Working directory for exoplanet fitting and modelling.


# Setup

## Environment

Follow the steps below (preferably in your `work` directory on aci. First create a [virtual environment](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/), and then activate it.

```
module load anaconda3
conda create -p=/storage/work/[INSERT YOUR USERNAME HERE]/work/[INSERT VENV NAME HERE] python=3.8
conda activate /storage/work/[INSERT YOUR USERNAME HERE]/work/[INSERT VENV NAME HERE]
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
You need to have a `DataParentDirectory` with a separate directory for each star (e.g. `TOI-1728`, `TOI-3629`, etc.), and then point to this path in 
`Code\Config.py`. The scripts will refer to this path for the photometry dataset, RV dataset, and the config file corresponding to each star.

```
pyexofits
|   Code


DataParentDirectory # Point to this path in Code/Config.py
|   Star1Name   RV_timeseries.csv
|   |   Photometry
|   |   |   Photometry_timeseries1.csv
|   |   |   Photometry_timeseries2.csv
|   |   Star1Name_config.txt
|   |   RV_timeseries.csv
|   Star2Name
....

```
