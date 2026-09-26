"""
xo_Astrometry.py -- Bayesian fits to Gaia DR4 pre-release epoch (along-scan)
astrometry for one source, optionally jointly with radial velocities.

Overview
--------
The script reads the per-CCD along-scan (AL) abscissae for one ``source_id``
from the Gaia DR4 pre-release epoch-astrometry VOTable and fits them with
PyMC3 / exoplanet. It always fits a 5-parameter single-star model first.
Depending on ``--Analysis`` it then builds a second model with a Keplerian
photocentre orbit, optionally joined to an RV model, seeded from the
single-star MAP solution. The final model is MAP-optimized, sampled with
NUTS, and summarized in a multi-page PDF plus csv and png outputs.

Analysis modes
--------------
``Single``
    Position offsets (dRA, dDec), proper motions (PMRA, PMDec), parallax,
    and an along-scan jitter term.
``Binary``
    The above plus a photocentre orbit in Campbell elements (P, a0, e,
    omega, Omega, i, T_peri), with the Thiele-Innes constants (A, B, F, G)
    recorded as derived quantities. By default the orbit is seeded from the
    Thiele-Innes grid search in ``xo_utils.scan_orbit_init``.
``Binary+RV``
    The above plus a joint RV model sharing (P, T_peri, e, omega) with the
    astrometric orbit. The RV semi-amplitude K is free rather than derived
    from a0, and a per-instrument offset and jitter and a shared quadratic
    (optionally linear) systemic trend are fit alongside.

Only the AL direction enters the likelihood; the across-scan (AC)
measurements are not used.

Conventions
-----------
Scan geometry
    ``Theta`` is the Gaia scan position angle in radians. The AL unit
    vector is ``(sin Theta, cos Theta)`` in (alpha*, delta), so the modelled
    abscissa is

        w = sin(Theta) * (dRA  + PMRA  * dt)     <- East  term
          + cos(Theta) * (dDec + PMDec * dt)     <- North term
          + Parallax * ParallaxFactorAl  [+ orbit projected onto AL]

Epochs
    ``dt`` is Julian years since J2017.5, the DR4 design-matrix reference
    epoch. The astrometric period, ``tperi`` and the orbit grid all live on
    that axis. RV times are BJD and are mapped onto it on the fly with
    ``(t - JD_REF_DR4) / 365.25``.

Time scale
    Gaia epoch times are TCB. They differ from the TDB scale RVs are usually
    quoted in by ~24 s with a slowly growing offset. No TCB-to-TDB
    conversion is applied.

Units
    Parallax, position offsets, a0 and the Thiele-Innes constants in mas;
    proper motions in mas/yr; the period in years internally (``period`` is
    reported in days); RVs, K, offsets and jitters in m/s; the RV trend
    terms in m/s/yr and m/s/yr^2.

The photocentre orbit
    The orbit is built as ``xo.orbits.KeplerianOrbit(..., a=-a0_mas)``.
    exoplanet places a companion at minus the semi-major axis, so the
    negative sign makes ``get_planet_position`` return a displacement of
    +a0 -- the photocentre, which for a faint companion moves with the
    primary. It returns ``(x, y, z)`` with x = North (Dec offset) and
    y = East (RA offset), which is why ``w_orb = y*sin(Theta) +
    x*cos(Theta)``. exoplanet reads ``a`` as R_sun and ``period`` as days and
    derives a total mass from them; with these inputs that mass is
    meaningless (and negative), but none of the orbit methods used here
    depend on it.

Angle degeneracy
    The sky orbit is unchanged under (omega, Omega) -> (omega + 180 deg,
    Omega + 180 deg), so in ``Binary`` mode those two angles are only
    defined up to that joint flip. RVs break the degeneracy in
    ``Binary+RV`` -- see the note at the RV model on how the sign used
    there fixes which of the two solutions is reported.

Command-line usage
------------------
    python xo_Astrometry.py --SOURCE_ID 1457486023639239296 \\
                            --StarName Gaia4 --Analysis Single

    python xo_Astrometry.py --SOURCE_ID 1457486023639239296 \\
                            --StarName Gaia4 --Analysis Binary

    python xo_Astrometry.py --SOURCE_ID 1457486023639239296 \\
                            --StarName Gaia4 --Analysis Binary+RV \\
                            --rv-csv /path/to/rvs.csv

Any argument left out falls back to the defaults set near the top of the
script.

Inputs
------
``<DataParentDirectory>/GAIA_DR4_PRERELEASE_EPOCH_ASTROMETRY_RAW.xml``
    The epoch-astrometry VOTable. ``DataParentDirectory`` is set in the
    local ``Config`` module.
``--rv-csv``
    Required for ``Binary+RV``. One csv holding every instrument, with
    columns ``bjd``, ``rv`` [m/s], ``e_rv`` [m/s] and ``inst`` (instrument
    label), one row per measurement. An optional boolean ``Use`` column is
    honoured by ``xo_utils.FilterRVData``.

Outputs
-------
Written to ``<DataParentDirectory>/<StarName>/Photometry/<RunName>/``, 

    ``Plots_<RunName>.pdf``        orbit-grid initialization, MAP and MCMC
                                   along-scan and sky plots, the posterior
                                   summary table, the trace plot, and the
                                   RV figures when RVs are fit
    ``TraceSummary.png``           the posterior summary table on its own
    ``ChainSummary_<RunName>.csv`` the same table as csv
    ``TraceFigure.png``            the trace plot on its own
    ``MCMC_Samples.csv``           posterior samples of the summary variables
    ``CornerPlot.png``             corner plot; written as ``CornerPlot.pdf``
                                   instead if the first attempt fails and is
                                   retried without NaN columns

Dependencies
------------
Third-party: numpy, matplotlib, pymc3, pymc3_ext, exoplanet, theano, corner,
and arviz (used by ``pm.summary`` / ``pm.traceplot``). pandas is imported but
not used directly.

Local modules shipped alongside this file: ``xo_utils`` (data loading, the
orbit-grid initializer, RV reading), ``xo_ResultPlots`` (all figures), and
``Config`` (defines ``DataParentDirectory``).

References
----------
The single-star along-scan model follows GAIA-C3-TN-LU-LL-061, available
from the public DPAC documents page:
https://www.cosmos.esa.int/web/gaia/public-dpac-documents
"""


# python xo_Astrometry.py --StarName "Gaia4" --SOURCE_ID 1457486023639239296 
#     --Analysis "Binary+RV" 
#     --rv-csv "/resnick/groups/carnegie_poc/skanodia/RunExoplanetRVs/Data/GaiaTesting/data/Gaia4b_GummiHARPSN_HPF2026.csv"


import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os, sys
import numpy as np
import pandas as pd

import exoplanet as xo
import pymc3 as pm
import theano.tensor as tt
import pymc3_ext as pmx
import corner

import datetime
import argparse

try:
	pwd = os.path.dirname(os.path.abspath(__file__))
except:
	pwd = r'/home/skanodia/resgroupdir/pyGaiafits/Code'

print(pwd)
sys.path.append(pwd)

from xo_utils import FilterRVData, load_epoch_astrometry, scan_orbit_init
from xo_ResultPlots import PlotRVCurve, PlotAstroAL, PlotAstroSky
from Config import DataParentDirectory


DR4_DESIGN_MATRIX_REF_JYEAR = 2017.5
JD_J2000 = 2451545.0
JD_REF_DR4 = JD_J2000 + (DR4_DESIGN_MATRIX_REF_JYEAR - 2000.0) * 365.25

##############################
# Note that Gaia time values are in TCB standard (Barycentric Coordinate Time), which is different from TDB.
# The difference between the two is ~ 24 seconds and is linearly increasing.

def SingleStarAstrometry(Data, dt, dRA, dDec, PMRA, PMDec, Parallax):
    """
    Modelled along-scan (AL) abscissa for a 5-parameter single-star solution.

    Equations based on 
    GAIA-C3-TN-LU-LL-061 available at `Public DPAC documents <https://www.cosmos.esa.int/web/gaia/public-dpac-documents>`__.

    The AL direction is the unit vector (sin Theta, cos Theta) in
    (alpha*, delta), so the East (RA) terms are projected with sin(Theta)
    and the North (Dec) terms with cos(Theta).

    Parameters
    ----------
    Data : pandas.DataFrame
        Epoch astrometry from load_epoch_astrometry. Only the Theta
        (scan position angle, radians) and ParallaxFactorAl columns are
        read here.
    dt : array_like
        Epoch times in Julian years relative to J2017.5 -- normally the
        'dt' column of the same table. Passed separately so the model can
        be evaluated on a different time grid.
    dRA, dDec : float or theano tensor
        Offsets from the catalogue reference position, mas. dRA is the
        great-circle offset, i.e. already multiplied by cos(dec).
    PMRA, PMDec : float or theano tensor
        Proper motions, mas/yr; PMRA is mu_alpha* = mu_alpha cos(dec).
    Parallax : float or theano tensor
        Parallax, mas.

    Returns
    -------
    w : array_like or theano tensor
        Modelled AL abscissa in mas, one entry per row of Data. Symbolic
        when the parameters are PyMC3 variables, a plain array for floats.
    """
    
    Theta = Data['Theta'].values

    # Along scan coordinate
    w = np.sin(Theta)*(dRA + PMRA*dt) + np.cos(Theta)*(dDec + PMDec*dt) + Parallax*Data['ParallaxFactorAl']
    #       ^ East term                      ^ North term    

    return w

# Input VOTable of raw epoch astrometry, anchored on DataParentDirectory
# from Config. For Gaia DR4, would replace this and the loading function with DR4 format.
XML_Path = os.path.join(DataParentDirectory, "GAIA_DR4_PRERELEASE_EPOCH_ASTROMETRY_RAW.xml")



# ----------------------------------------------------------------------
# Defaults. SOURCE_ID, StarName, FitBinary and FitRV are overridden by the
# command-line arguments below whenever those are given (FitBinary and
# FitRV through --Analysis). The rest are only changed by editing here:
#   GuessOrbit : seed the orbit from scan_orbit_init's Thiele-Innes grid
#                search instead of broad uninformative priors.
#   LinRVTrend : True fits a linear systemic RV trend, False (default)
#                linear + quadratic. Only used when FitRV is on.
#   RunName    : descriptive middle of the output name; it is a fixed
#                string, so update it by hand if the settings change.
#   Nchains    : parallel NUTS chains, also the number of cores used.
# The first StarName assignment is immediately overwritten by the second.
# ----------------------------------------------------------------------
# Single Star
SOURCE_ID = 1457486023639239296 # Gaia-4

StarName = SOURCE_ID
StarName = 'Gaia4'
FitBinary = True
FitRV = False
GuessOrbit = True

LinRVTrend = False

    
RunName = 'AllRVs_QuadTrend_UseEverything'
Nchains = 3



# ----------------------------------------------------------------------
# Command-line interface. Every argument is optional; --rv-csv becomes
# required once --Analysis Binary+RV is chosen. Note the exit message
# below still spells the instrument column 'Inst', while the help text
# and the check in the RV reader use 'inst'.
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Run Astrometric Fit")

# Add arguments to the parser
parser.add_argument("--StarName", required=False, type=str, default=None, help="StarName")
parser.add_argument("--SOURCE_ID", required=False, type=int, default=None, help="SOURCE_ID")
parser.add_argument("--Analysis", choices=["Single", "Binary", "Binary+RV"], default=None)
parser.add_argument("--rv-csv", required=False, type=str, default=None,
	help="Path to a combined RV csv (columns: bjd, rv, e_rv, inst) required for --Analysis Binary+RV")

args = vars(parser.parse_args())

if args['StarName'] is not None:
	StarName = args['StarName']

if args['SOURCE_ID'] is not None:
	SOURCE_ID = args['SOURCE_ID']
	print("Using argparse SOURCE_ID = "+ str(SOURCE_ID))

if args['Analysis'] is not None:
    if args['Analysis'] == 'Single': 
        FitBinary = False
        FitRV = False
    elif args['Analysis'] == 'Binary': 
        FitBinary = True        
        FitRV = False
    elif args['Analysis'] == 'Binary+RV': 
        FitBinary = True        
        FitRV = True

if FitRV and args['rv_csv'] is None:
	sys.exit("--Analysis Binary+RV requires --rv-csv (columns: bjd, rv, e_rv, Inst)")


########################################
########################################
if FitRV: Prefix = 'BinaryRVFit'
elif FitBinary: Prefix = 'BinaryFit'
else: Prefix = 'SingleFit'

RunName = Prefix + '_' + RunName + '_' +  StarName

print("RunName = ", RunName)


# Create a new ResultDirectory in DataDirectory based on the RunName.
# os.mkdir creates only the final level, so
# <DataParentDirectory>/<StarName>/Photometry/ must already exist or this
# raises FileNotFoundError. An existing directory is reused, and files
# from an earlier run with the same RunName are overwritten.
DataDirectory = os.path.join(DataParentDirectory, StarName)
ResultDirectory = os.path.join(DataDirectory, 'Photometry', RunName)

if os.path.exists(ResultDirectory):
	print(ResultDirectory + " already exists")
else:
	os.mkdir(ResultDirectory)

########################################
########################################

AstroDataset = load_epoch_astrometry(xml_path=XML_Path, source_id=SOURCE_ID, only_used_by_agis=True)
# Sorting by dt is load-bearing, not cosmetic: PlotAstroAL pairs model
# and data element-wise in its residual panel and is only correct for a
# time-ordered table (see its docstring). reset_index() without drop=True
# also keeps the loader's original row number as an 'index' column.
AstroDataset = AstroDataset.sort_values("dt").reset_index()

Theta = AstroDataset['Theta'].values
# Parallax factors rotated from (AL, AC) into (RA, Dec). Kept for
# inspection; the fit itself uses ParallaxFactorAl directly and never
# reads these. They are all NaN if the input table has no
# parallax_factor_ac column.
ParallaxFactorRA = AstroDataset['ParallaxFactorAl'].values * np.sin(Theta) + AstroDataset['ParallaxFactorAc'].values * np.cos(Theta)
ParallaxFactorDec = AstroDataset['ParallaxFactorAl'].values * np.cos(Theta) - AstroDataset['ParallaxFactorAc'].values * np.sin(Theta)    

# Thiele-Innes grid search over trial periods, eccentricities and
# periastron times. Returns starting Campbell elements (P_days, a0, e,
# omega, Omega, cosi, phase) and a diagnostic figure for the PDF.
# P_min=None / P_max=None select its defaults: 10 days and twice the
# observing baseline.
if FitBinary & GuessOrbit:
    InitBinaryOrbitPriors, OrbitGuessFig  = scan_orbit_init(data=AstroDataset, map_soln=None, P_min=None, P_max=None , oversample=10.0, top_k=5, MakePlot=True)


########################################################################
########################################################################
# Read in RVs, one instrument at a time, keyed off the 'inst' column.
# Julian year J2017.5 in JD, matching the DR4_DESIGN_MATRIX_REF_JYEAR convention
# used for tperi/Period elsewhere in this script (dt = years since this epoch).


if FitRV:

    print("Reading in RVs from " + args['rv_csv'])
    AllRVData = FilterRVData(args['rv_csv'])
    if 'inst' not in AllRVData.columns:
        sys.exit("RV csv must have an 'inst' column identifying the instrument per row")
    
    RVDict = {}
    RVInstruments = []
    
    X_RVs = []
    Y_RVs = []
    Yerr_RVs = []
    InstID = []
    InstIndex = 0    

    RVInstruments = sorted(AllRVData['inst'].unique())
    for RVInst in RVInstruments:
        print("Reading in RV Instrument " + RVInst)
        RVData = AllRVData[AllRVData['inst'] == RVInst]
        # Time, RV, Error in RV
        x_rv = np.array(RVData.bjd)
        y_rv = np.array(RVData.rv) # m/s
        yerr_rv = np.array(RVData.e_rv)
        
        RVDict['x_rv_'+RVInst] = x_rv
        RVDict['y_rv_'+RVInst] = y_rv
        RVDict['yerr_rv_'+RVInst] = yerr_rv
    
        X_RVs.append(x_rv)
        Y_RVs.append(y_rv)
        Yerr_RVs.append(yerr_rv)
        InstID.append(np.repeat(InstIndex, len(x_rv)))
        
        # Increment the Instrument Index
        InstIndex += 1
    
    ## Sort and clean the RV arrays
    X_RVs = np.concatenate(X_RVs)
    Y_RVs = np.concatenate(Y_RVs)
    Yerr_RVs = np.concatenate(Yerr_RVs)
    InstID = np.concatenate(InstID)
    
    inds = np.argsort(X_RVs)
    X_RVs =  np.ascontiguousarray(X_RVs[inds], dtype=float)
    Y_RVs =  np.ascontiguousarray(Y_RVs[inds], dtype=float)
    Yerr_RVs =  np.ascontiguousarray(Yerr_RVs[inds], dtype=float)
    InstID =  np.ascontiguousarray(InstID[inds], dtype=int)
    
    RVDict['X_RVs'] = X_RVs
    RVDict['Y_RVs'] = Y_RVs
    RVDict['Yerr_RVs'] = Yerr_RVs
    RVDict['InstID'] = InstID
    RVDict['RVInstruments'] = RVInstruments
            
    # Fine BJD grid spanning the RVs plus 10 days either side, on which the
    # RV model is evaluated for plotting only.
    t_rv = np.linspace(X_RVs.min()-10, X_RVs.max()+10, 5000)
    RVDict['t_rv'] = t_rv

    # Reference epoch for the RV background trend (linear + quadratic), in BJD,
    # anchored at the mean RV time across all instruments to reduce the
    # correlation between the trend terms and the per-instrument offsets.
    X_RVs_Ref = np.mean(X_RVs)
    print("RV instruments found: " + ", ".join("%s (n=%d)" % (i, len(RVDict['x_rv_' + i])) for i in RVInstruments))

########################################################################
########################################################################
# ----------------------------------------------------------------------
# Stage 1: single-star (5-parameter) fit.
#
# Runs in every mode. For Single it is the final model; otherwise only its
# MAP solution is used, to seed the test values of the shared parameters
# in stage 2, and the model itself is discarded.
#
# The likelihood is a zero-mean Normal on the residuals, equivalent to a
# Normal on w but letting the jitter enter the width directly.
# exp(logAstroJitter) is an excess AL scatter in mas, added in quadrature
# to w_err.
# ----------------------------------------------------------------------
print("Starting with single star fit")

with pm.Model() as model:
    
    Parallax = pm.Uniform("Parallax", lower=1e-3, upper=100, testval=10.0)

    # Stellar proper motion
    PMRA = pm.Normal("PMRA", mu=0.0, sd=500.0) # ra*cos(dec), mas/yr
    PMDec = pm.Normal("PMDec", mu=0.0, sd=500.0)
    
    # Stellar position offest
    dRA = pm.Normal("dRA", mu=0.0, sd=10.0) # ra*cos(dec), mas
    dDec = pm.Normal("dDec", mu=0.0, sd=10.0)

    logAstroJitter = pm.Normal("logAstroJitter", mu=0.0, sigma=1.0)

    # Define the Single Star Model
    w_SingleModel = SingleStarAstrometry(AstroDataset, AstroDataset['dt'].values,  dRA, dDec, PMRA, PMDec, Parallax)
    w_SSModel = pm.Deterministic("w_SSModel", w_SingleModel) 
    Residuals = AstroDataset['w'].values - w_SSModel

    SSAstrometryLogLike = pm.Normal("SSAstrometryObs", 
        mu=0, 
        sd=tt.sqrt(tt.exp(logAstroJitter)**2 + AstroDataset['w_err'].values**2),
        observed=Residuals)

    #### Things needed for plotting ####
    # Proper-motion track on the sky (parallax deliberately left out), and
    # the observations reconstructed onto it by projecting each AL residual
    # back along its own scan direction. Only the AL component is
    # measured, so these points are a visualization, not 2D positions.
    RA_model_SS = pm.Deterministic("RA_model_SS",
        dRA + (PMRA * AstroDataset['dt'].values))   
    Dec_model_SS = pm.Deterministic("Dec_model_SS",
        dDec + (PMDec * AstroDataset['dt'].values))

    RA_data_SS = pm.Deterministic("RA_data_SS", RA_model_SS + Residuals* np.sin(Theta))
    Dec_data_SS = pm.Deterministic("Dec_data_SS", Dec_model_SS + Residuals* np.cos(Theta))    

    # Double check that everything looks good - we shouldn't see any NaNs!
    print(model.check_test_point())
    map_soln = model.test_point
    map_soln = pmx.optimize(map_soln)   # final joint stage, everything moves together
    print("====\nFinished MAP optimization\n====")

print(""" Finished Single Star Fit """)

print(model.logp(map_soln))
for k, v in sorted(map_soln.items()):
    if not k.endswith("__") and np.ndim(v) == 0:
        print("  %-14s % .6g" % (k, float(v)))

########################################################################
########################################################################

if FitBinary:

    # ------------------------------------------------------------------
    # Stage 2: single star + photocentre orbit (+ RVs).
    #
    # A fresh pm.Model is opened and the name `model` is rebound to it, so
    # everything after this block refers to the binary model. The
    # astrometric parameters are re-declared with test values taken from
    # the stage-1 MAP. Note the Parallax prior here extends to 1000 mas,
    # against 100 mas in stage 1.
    # ------------------------------------------------------------------
    print("Now to binary star fit" + (" + RV" if FitRV else ""))    
    with pm.Model() as model:
        
        Parallax = pm.Uniform("Parallax", lower=1e-3, upper=1000, testval=map_soln['Parallax']) # mas
    
        # Stellar proper motion
        PMRA = pm.Normal("PMRA", mu=0.0, sd=500.0, testval=map_soln['PMRA']) # ra*cos(dec), mas/yr
        PMDec = pm.Normal("PMDec", mu=0.0, sd=500.0, testval=map_soln['PMDec'])
    
        # Stellar position offest
        dRA = pm.Normal("dRA", mu=0.0, sd=10.0, testval=map_soln['dRA']) # ra*cos(dec), mas
        dDec = pm.Normal("dDec", mu=0.0, sd=10.0, testval=map_soln['dDec'])
    
        logAstroJitter = pm.Normal("logAstroJitter", mu=0.0, sigma=1.0, testval=map_soln['logAstroJitter'])
            
        # Define the Single Star Model
        w_SingleModel = SingleStarAstrometry(AstroDataset, AstroDataset['dt'].values,  dRA, dDec, PMRA, PMDec, Parallax)
        w_SSModel = pm.Deterministic("w_SSModel", w_SingleModel) 
        Residuals = AstroDataset['w'].values - w_SSModel
    
        #### Things needed for plotting ####
        RA_model_SS = pm.Deterministic("RA_model_SS",
            dRA + (PMRA * AstroDataset['dt'].values))   
        Dec_model_SS = pm.Deterministic("Dec_model_SS",
            dDec + (PMDec * AstroDataset['dt'].values))
    
        RA_data_SS = pm.Deterministic("RA_data_SS", RA_model_SS + Residuals* np.sin(Theta))
        Dec_data_SS = pm.Deterministic("Dec_data_SS", Dec_model_SS + Residuals* np.cos(Theta))
        
        ############################################        
        ############################################          
        # Always true here -- this whole block already sits inside
        # `if FitBinary:` -- so the test is redundant.
        if FitBinary:
        
            # # Companion mass
            # LogMp = pm.Uniform('LogMp', lower=np.log(1e-3), upper=np.log(1e3), testval=np.log(1)) # solar masses
            # Mp = pm.Deterministic('Mp', tt.exp(LogMp))
            
            # Binary Modelling 
            if not GuessOrbit:
                logPeriod = pm.Uniform('logPeriod', lower=np.log(10/365.25), upper=np.log(1e4/365.25), testval=np.log(1e3/365.25)) # years
                loga0 = pm.Uniform('loga0', lower=np.log(1), upper=np.log(1000), testval=np.log(1e2)) # mas
                cosi = pm.Uniform('cosi', lower=-1., upper=1., testval=0) 
    
                # sqrt(e)*cos(omega), sqrt(e)*sin(omega)
                ecs = pmx.UnitDisk("ecs", testval=np.array([0.5, 0.5]))
                Omega = pmx.Angle("Omega", testval=60*np.pi/180)
                phase = pmx.Angle("phase", testval=60*np.pi/180)
            
            # Seeded priors. logPeriod is Normal in log-years with sd 0.7
            # (roughly a factor of 2 in P); loga0 is Normal in log-mas with
            # sd 2.0, i.e. essentially only a starting point.
            else:
                logPeriod = pm.Normal('logPeriod', mu=np.log(InitBinaryOrbitPriors['P_days']/365.25), sd=0.7, 
                                      testval=np.log(InitBinaryOrbitPriors['P_days']/365.25)) # years
                loga0 = pm.Normal("loga0", mu=np.log(InitBinaryOrbitPriors["a0"]), sd=2.0,
                              testval=np.log(InitBinaryOrbitPriors["a0"]))
                cosi = pm.Uniform('cosi', lower=-1., upper=1., testval=InitBinaryOrbitPriors['cosi']) 
                # ecs = sqrt(e) * (cos omega, sin omega). A circular grid
                # solution is nudged to sqrt(e) = 0.05 so the sampler does not
                # start at the disk centre, where omega is undefined.
                se = np.sqrt(InitBinaryOrbitPriors["e"]) if InitBinaryOrbitPriors["e"] > 0 else 0.05
                ecs = pmx.UnitDisk("ecs", testval=np.array(
                    [se * np.cos(InitBinaryOrbitPriors["omega"]), se * np.sin(InitBinaryOrbitPriors["omega"])]))
                Omega = pmx.Angle("Omega", testval=InitBinaryOrbitPriors["Omega"])
                phase = pmx.Angle("phase", testval=InitBinaryOrbitPriors["phase"])
        
            
            # Derived Campbell elements. ecc = |ecs|^2 and
            # omega = atan2(ecs_y, ecs_x); sampling in the unit disk avoids the
            # e-omega degeneracy at low e. tperi is in years since J2017.5, on
            # the same axis as dt.
            period_years = pm.Deterministic('period_years', tt.exp(logPeriod))
            period = pm.Deterministic('period', period_years * 365.25)
            a0_mas = pm.Deterministic("a0_mas", tt.exp(loga0))                    # mas  
            incl = pm.Deterministic('incl', tt.arccos(cosi))
            pm.Deterministic("incl_deg", incl * 180.0 / np.pi)
        
            ecc = pm.Deterministic("ecc", tt.sum(ecs ** 2))
            omega = pm.Deterministic("omega", tt.arctan2(ecs[1], ecs[0]))
            pm.Deterministic("omega_deg", omega * 180.0 / np.pi) # Argument of periastron
            pm.Deterministic("Omega_deg", Omega * 180.0 / np.pi) # Longitude of Ascending Node
            tperi = pm.Deterministic("tperi", period_years * phase / (2.0 * np.pi))  # yr, J2017.5
            
            # Thiele-Innes constants in mas, tracked for comparison with
            # catalogue-style (A, B, F, G) solutions. They are invariant under
            # the joint (omega, Omega) + 180 deg flip described in the module
            # docstring, so they are well defined even where those two angles
            # are not.
            cw, sw = tt.cos(omega), tt.sin(omega)
            cO, sO = tt.cos(Omega), tt.sin(Omega)
            A = pm.Deterministic("A_TI", a0_mas * (cw * cO - sw * sO * cosi))
            B = pm.Deterministic("B_TI", a0_mas * (cw * sO + sw * cO * cosi))
            F = pm.Deterministic("F_TI", -a0_mas * (sw * cO + cw * sO * cosi))
            G = pm.Deterministic("G_TI", -a0_mas * (sw * sO - cw * cO * cosi))    
        
            # Define the orbit. a = -a0_mas: exoplanet places the companion at
            # minus the semi-major axis.
            # Working in mas means no stellar mass or distance enters -- AL
            # astrometry constrains only the angular orbit.
            # Define the orbit
            orbit = xo.orbits.KeplerianOrbit(period=period_years, a=-a0_mas,
                                             t_periastron=tperi, 
                                             incl=incl, Omega=Omega,
                                             ecc=ecc, omega=omega, )
                                             # m_star=mstar, r_star=rstar,
                                             # m_planet=mp)

            # exoplanet's reference (conjunction) time, converted to BJD. Not
            # used anywhere downstream or included in the summary.
            t0 = pm.Deterministic("t0", orbit.t0*365.25 + JD_REF_DR4)
        
            x_orb, y_orb, _ = orbit.get_planet_position(AstroDataset['dt'].values)
            # x = North (Dec offset)
            # y = East (RA offset)
        
            # Project onto the scan direction, matching SingleStarAstrometry:
            # East (y) with sin(Theta), North (x) with cos(Theta).
            w_orb = pm.Deterministic("w_orb", y_orb * tt.sin(Theta) + x_orb * tt.cos(Theta))
            w_BSModel = pm.Deterministic("w_BSModel", w_SSModel+w_orb)
            Residuals = AstroDataset['w'].values - w_BSModel

            #### Things needed for plotting (binary) ####
            RA_model_BS = pm.Deterministic("RA_model_BS", RA_model_SS + y_orb)
            Dec_model_BS = pm.Deterministic("Dec_model_BS", Dec_model_SS + x_orb)
            RA_data_BS = pm.Deterministic("RA_data_BS", RA_model_BS + Residuals * tt.sin(Theta))
            Dec_data_BS = pm.Deterministic("Dec_data_BS", Dec_model_BS + Residuals * tt.cos(Theta))

            # One full orbit sampled uniformly in time from tperi, for drawing
            # the relative orbit on the sky plot.
            PhaseGrid = np.linspace(0.0, 2.0 * np.pi, 300)
            t_orbit_grid = tperi + PhaseGrid * period_years / (2.0 * np.pi)
            x_orb_grid, y_orb_grid, _ = orbit.get_planet_position(t_orbit_grid)
            RA_orbit_grid = pm.Deterministic("RA_orbit_grid", y_orb_grid)
            Dec_orbit_grid = pm.Deterministic("Dec_orbit_grid", x_orb_grid)


            ############################################
            ############################################
            # Joint RV fit: shares (Period, tperi, ecc, omega) with the
            # astrometric orbit above. get_true_anomaly only depends on
            # those four (not on 'a'), so it's safe to reuse `orbit` here
            # even though its 'a' was set to -a0 for the astrometric fit --
            # K_RV is its own free parameter, not derived from a0.
            if FitRV:
                
                logK_RV = pm.Uniform("logK_RV", lower=np.log(1e-2), upper=np.log(1e6),
                    testval=np.log(500.0), shape=1)  # m/s
                K_RV = pm.Deterministic("K_RV", tt.exp(logK_RV))
                K0 = pm.Deterministic("K0", K_RV)

                # `K_pred = 2 pi a0_au sin(i) / (P sqrt(1-e^2)) * 4740.470446` [m/s]

                
                # Shared systemic RV trend: linear + quadratic in years,
                # relative to X_RVs_Ref (mean RV time, in BJD)
                # m/s/year & m/s/year2

                if LinRVTrend: RVTrend = pm.Normal("RVTrend", mu=0.0, sd=100.0, shape=1, testval=0.0)             
                else: RVTrend = pm.Normal("RVTrend", mu=0.0, sd=100.0, shape=2, testval=np.array([0.0, 0.0])) 
                
                # Per instrument parameters
                # One systemic offset and one jitter per instrument. Each
                # offset prior is centred on that instrument's median RV;
                # the jitter is HalfNormal with a 50 m/s scale.
                RVOffset = pm.Normal(
                    "RVOffset",
                    mu=np.array([np.median(Y_RVs[InstID == i]) for i in range(len(RVInstruments))]),
                    sigma=2500.,
                    shape=len(RVInstruments),
                )
                RVJitter = pm.HalfNormal("RVJitter", sigma=50, shape=len(RVInstruments))
                # logRVJitter = pm.Uniform("logRVJitter", -3, 3,shape=len(RVInstruments))
                # RVJitter = pm.Deterministic("RVJitter", tt.exp(logRVJitter))
                
                # Compute the RV offset and jitter for each data point depending on its instrument
                # Scatter each instrument's offset and total variance onto the
                # measurements via InstID. RVDiag is a variance -- formal error
                # squared plus that instrument's jitter squared.
                RVMean = tt.zeros(len(X_RVs))
                RVDiag = tt.zeros(len(X_RVs))
                for i in range(len(RVInstruments)):
                    RVMean += RVOffset[i] * (InstID == i)
                    RVDiag += (Yerr_RVs ** 2 + RVJitter[i] ** 2) * (InstID == i)
                pm.Deterministic("RVMean", RVMean)
                pm.Deterministic("RVDiag", RVDiag)
                OffsetSubtracted = Y_RVs - RVMean

                t_ref = (X_RVs_Ref- JD_REF_DR4)/365.25
                
                # And a function for computing the full RV model
                def get_rv_model(t, name=''):
                    ''' time t is expected in the same units as that passed for the orbit '''
                    
                    # Sign convention -- read before interpreting omega/Omega.
                    # In exoplanet's frame (z towards the observer), the body
                    # that get_planet_position places at +a0 -- the photocentre
                    # here -- has line-of-sight velocity
                    # +get_radial_velocity(K). The minus sign therefore models
                    # RV motion opposite to the photocentre. 
                    vrad = - orbit.get_radial_velocity(t, K=K_RV)
                    pm.Deterministic("vrad" + name, vrad)

                    if LinRVTrend: bkg = pm.Deterministic("bkg" + name, RVTrend[0]*(t-t_ref))
                    else: bkg = pm.Deterministic("bkg" + name, RVTrend[0]*(t-t_ref)  + RVTrend[1]*((t-t_ref)**2))
                    
                    return pm.Deterministic("rv_model" + name, vrad + bkg)
                
                # Define the RVs at the observed times
                rv_model = get_rv_model((X_RVs - JD_REF_DR4)/365.25)
                
                # Also define the model on a fine grid as computed above (for plotting)
                rv_model_pred = get_rv_model((t_rv - JD_REF_DR4)/365.25, name="_pred")
                
                # RV likelihood on the offset-subtracted velocities, with each
                # point's width including its instrument's jitter.
                rv_loglike = pm.Normal("rv_obs", mu=rv_model, sd=tt.sqrt(RVDiag), observed=OffsetSubtracted)
            
            ############################################
            ############################################

        
        # Inside the FitBinary block this test is always False, so only the
        # binary-star likelihood branch ever runs.
        if not FitBinary:
            SSAstrometryLogLike = pm.Normal("SSAstrometryObs", 
                mu=0, 
                sd=tt.sqrt(tt.exp(logAstroJitter)**2 + AstroDataset['w_err'].values**2),
                observed=Residuals)
        else:
            BSAstrometryLogLike = pm.Normal("BSAstrometryObs", 
                mu=0, 
                sd=tt.sqrt(tt.exp(logAstroJitter)**2 + AstroDataset['w_err'].values**2),
                observed=Residuals)
            
        # Double check that everything looks good - we shouldn't see any NaNs!
        print(model.check_test_point())
        # Two-stage MAP: move only the astrometric parameters with the orbit
        # held at its seeded values, then release everything. Far more robust
        # than optimizing the full set from the grid-search start in one go.
        print("====\nStarting MAP optimization\n====")        
        map_soln = model.test_point
        map_soln = pmx.optimize(map_soln, vars=[Parallax,  PMRA, PMDec, dRA, dDec, logAstroJitter])
        map_soln = pmx.optimize(map_soln)   # final joint stage, everything moves together
        print("====\nFinished MAP optimization\n====")
            
# Log-probability and scalar MAP values of the final model. In Single mode
# this repeats the stage-1 printout above.
print(model.logp(map_soln))
for k, v in sorted(map_soln.items()):
    if not k.endswith("__") and np.ndim(v) == 0:
        print("  %-14s % .6g" % (k, float(v)))

########################################################################
########################################################################

# ----------------------------------------------------------------------
# Figures at the MAP solution. PlotAstroAL shows the AL abscissa (or, with
# OrbitKey, the isolated orbit signal) against time with residuals below;
# PlotAstroSky shows the reconstructed sky motion, the orbit, and the
# residuals of both models. Only the figures appended to the PDF at the
# end are kept; SavePlot is off, so none is written as a png here.
# ----------------------------------------------------------------------
############ MAP ############

SS_MAP_ALCentroidPlot = PlotAstroAL(AstroDataset, soln=map_soln, trace=None, outdir=ResultDirectory,
                                    label="Single Star MAP", ModelKey='w_SSModel')
if FitBinary:
    BS_MAP_ALCentroidPlot = PlotAstroAL(AstroDataset, soln=map_soln, trace=None, outdir=ResultDirectory,
                                        label="Non Single Star MAP", ModelKey='w_BSModel',
                                        OrbitKey='w_orb')

MAP_SkyPlot = PlotAstroSky(AstroDataset, soln=map_soln, trace=None, outdir=ResultDirectory,
                           label="MAP")

if FitBinary & FitRV:
    InitialRVModel = PlotRVCurve(RVDict=RVDict, RV_GP=False,
    		soln=map_soln, MakePlots=False, Title='MAP')

# ----------------------------------------------------------------------
# NUTS sampling via pymc3_ext, started at the MAP solution. target_accept
# is high because the orbital parameters are strongly correlated; raise
# tune/draws for poorly constrained orbits.
# ----------------------------------------------------------------------
print("==============\nStarting Posterior Estimation chains\n==============")

with model:
	trace = pmx.sample(
		tune=3000, # 3000
		draws=1500, # 1500
		start=map_soln,
		cores=Nchains,
		chains=Nchains,
		initial_accept=0.85,
		target_accept=0.95,
	)
print(datetime.datetime.now())
print("==============\nFinished Posterior Estimation\n==============")


############ MCMC ############

SS_MCMC_ALCentroidPlot = PlotAstroAL(AstroDataset, soln=None, trace=trace, outdir=ResultDirectory,
                                     label="Single Star MCMC", ModelKey='w_SSModel')
if FitBinary:
    BS_MCMC_ALCentroidPlot = PlotAstroAL(AstroDataset, soln=None, trace=trace, outdir=ResultDirectory,
                                         label="Non Single Star MCMC", ModelKey='w_BSModel',
                                         OrbitKey='w_orb')

MCMC_SkyPlot = PlotAstroSky(AstroDataset, soln=None, trace=trace, outdir=ResultDirectory,
                            label="MCMC")


########################################################################
# Parameters reported in the summary table, csv outputs, trace plot and
# corner plot. Built to match the model actually fit -- pm.summary raises
# on a variable that is not in the trace.
var_names = ['PMRA', 'PMDec', 'dRA', 'dDec', 'logAstroJitter', 'Parallax']

if FitBinary:
    var_names.append("logPeriod")
    var_names.append("period")
    var_names.append("period_years")    
    var_names.append("loga0")    
    var_names.append("a0_mas")        
    var_names.append("tperi")    
    var_names.append("ecc")    
    var_names.append("phase")     
    
    var_names.append("omega")    
    var_names.append("omega_deg")  
    
    var_names.append("Omega")    
    var_names.append("Omega_deg")    

    var_names.append("cosi")    
    var_names.append("incl_deg")  

    var_names.append("A_TI")  
    var_names.append("B_TI")  
    var_names.append("F_TI")  
    var_names.append("G_TI")  

if FitRV:
    var_names.append("logK_RV")
    var_names.append("K_RV")
    var_names.append("RVTrend")
    var_names.append("RVOffset")
    var_names.append("RVJitter")    
    
########################################################################

with model:
	df = pm.summary(
		trace, var_names=var_names, stat_funcs = {"median":np.median}
	)

df.reset_index(inplace=True)
df = df.rename(columns={'index': 'Parameter'})

# Render the summary DataFrame as a table figure, for the PDF and a png.
TracesSummary, ax = plt.subplots(figsize=(12,8))
ax.axis('tight')
ax.axis('off')
ax.set_title('Summary from {} chains'.format(trace.nchains))
the_table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
the_table.auto_set_font_size(False)
TracesSummary.savefig(os.path.join(ResultDirectory, 'TraceSummary.png'))
df.to_csv(os.path.join(ResultDirectory, 'ChainSummary_{}.csv'.format(RunName)))



with model:
	TracePlot = pm.traceplot(trace, var_names=var_names)
	TraceFigure = TracePlot[0][0].get_figure()
	TraceFigure.savefig(os.path.join(ResultDirectory, 'TraceFigure.png'))
	samples = pm.trace_to_dataframe(trace, varnames=var_names)
     
samples.to_csv(os.path.join(ResultDirectory, 'MCMC_Samples.csv'), index=False)


print("==============\nFinished Generating Trace Results\n==============")
########## Corner Plot #################
# First attempt uses every summary variable. If corner raises -- usually
# because a column is all or partly NaN -- the bare except retries with
# the NaN columns dropped. Note the retry saves CornerPlot.pdf where the
# first attempt saves CornerPlot.png, and the bare except will also
# swallow errors unrelated to NaNs. The corner plot is not added to the
# multi-page PDF.
try:
	CornerPlot = corner.corner(samples, quantiles=[0.16, 0.5, 0.84],
			show_titles=True, title_kwargs={"fontsize": 12}, use_math_text=True)
	CornerPlot.savefig(os.path.join(ResultDirectory, 'CornerPlot.png'), dpi=360)     
except:
	MCMCColumns = np.array(samples.columns)
	iii = [~ np.any(np.isnan(samples[c])) for c in MCMCColumns]
	NonNanColumns =  MCMCColumns[iii]
	print("NaN columns = "+ MCMCColumns[~np.array(iii)])
	CornerPlot = corner.corner(samples[NonNanColumns], quantiles=[0.16, 0.5, 0.84],
			show_titles=True, title_kwargs={"fontsize": 12}, use_math_text=True)
	CornerPlot.savefig(os.path.join(ResultDirectory, 'CornerPlot.pdf'), dpi=360)

print("==============\nFinished Generating Corner Plot \n==============")

########################################################################
########################################################################

if FitBinary & FitRV:
    MCMCPosterior = PlotRVCurve(RVDict=RVDict, RV_GP=False,
    		trace=trace, MakePlots=False, Title='MCMC', NPlanets=1)
    

########################################################################
########################################################################
# ----------------------------------------------------------------------
# Collect the figures into one multi-page PDF, in this order: orbit
# initialization, initial RV model, MAP AL plot, MAP sky plot, MCMC AL
# plot, MCMC sky plot, summary table, trace plot, RV posterior. The AL
# plot kept is the single-star one in Single mode and the orbit-isolated
# binary one otherwise.
# ----------------------------------------------------------------------
pp = PdfPages(os.path.join(ResultDirectory, 'Plots_{}.pdf'.format(RunName)))

if FitBinary & GuessOrbit: 
    pp.savefig(OrbitGuessFig)

if FitBinary & FitRV:     
    pp.savefig(InitialRVModel)

if not FitBinary: pp.savefig(SS_MAP_ALCentroidPlot)
else: pp.savefig(BS_MAP_ALCentroidPlot)
pp.savefig(MAP_SkyPlot)    

if not FitBinary: pp.savefig(SS_MCMC_ALCentroidPlot)
else: pp.savefig(BS_MCMC_ALCentroidPlot)
pp.savefig(MCMC_SkyPlot)    

# pp.savefig(OnSkyFigure)
pp.savefig(TracesSummary)
pp.savefig(TraceFigure)

if FitBinary & FitRV: 
    pp.savefig(MCMCPosterior)

pp.close()
print("==============\nSaved everything for {}\n==============".format(RunName))
########################################################################
########################################################################



