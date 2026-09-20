

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
	pwd = r'/home/skanodia/work/pyGaiafits/Code'

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
    Equations based on 
    GAIA-C3-TN-LU-LL-061 available at `Public DPAC documents <https://www.cosmos.esa.int/web/gaia/public-dpac-documents>`__.
    """
    
    Theta = Data['Theta'].values

    # Along scan coordinate
    w = np.sin(Theta)*(dRA + PMRA*dt) + np.cos(Theta)*(dDec + PMDec*dt) + Parallax*Data['ParallaxFactorAl']
    #       ^ East term                      ^ North term    

    return w

XML_Path = os.path.join(DataParentDirectory, "GAIA_DR4_PRERELEASE_EPOCH_ASTROMETRY_RAW.xml")

ResultDirectory = os.path.join(DataParentDirectory, 'GaiaTesting')
print(ResultDirectory)

# Single Star
SOURCE_ID = 1457486023639239296 # Gaia-4


StarName = SOURCE_ID
StarName = 'Gaia4'
FitBinary = True
FitRV = False
GuessOrbit = True

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

LinRVTrend = False

    
RunName = Prefix + '_AllRVs_QuadTrend_UseEverything_' + StarName

print("RunName = ", RunName)
########################################
########################################

Nchains = 3

AstroDataset = load_epoch_astrometry(xml_path=XML_Path, source_id=SOURCE_ID, only_used_by_agis=True)
AstroDataset = AstroDataset.sort_values("dt").reset_index()

Theta = AstroDataset['Theta'].values
ParallaxFactorRA = AstroDataset['ParallaxFactorAl'].values * np.sin(Theta) + AstroDataset['ParallaxFactorAc'].values * np.cos(Theta)
ParallaxFactorDec = AstroDataset['ParallaxFactorAl'].values * np.cos(Theta) - AstroDataset['ParallaxFactorAc'].values * np.sin(Theta)    

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
            
    t_rv = np.linspace(X_RVs.min()-10, X_RVs.max()+10, 5000)
    RVDict['t_rv'] = t_rv

    # Reference epoch for the RV background trend (linear + quadratic), in BJD,
    # anchored at the mean RV time across all instruments to reduce the
    # correlation between the trend terms and the per-instrument offsets.
    X_RVs_Ref = np.mean(X_RVs)
    print("RV instruments found: " + ", ".join("%s (n=%d)" % (i, len(RVDict['x_rv_' + i])) for i in RVInstruments))

########################################################################
########################################################################
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

    print("Now to binary star fit" + (" + RV" if FitRV else ""))    
    with pm.Model() as model:
        
        Parallax = pm.Uniform("Parallax", lower=1e-3, upper=1000, testval=map_soln['Parallax']) # mas
    
        # Stellar proper motion
        PMRA = pm.Normal("PMRA", mu=0.0, sd=500.0, testval=map_soln['PMRA']) # ra*cos(dec), arcsec/yr
        PMDec = pm.Normal("PMDec", mu=0.0, sd=500.0, testval=map_soln['PMDec'])
    
        # Stellar position offest
        dRA = pm.Normal("dRA", mu=0.0, sd=10.0, testval=map_soln['dRA']) # ra*cos(dec), arcsec
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
            
            else:
                logPeriod = pm.Normal('logPeriod', mu=np.log(InitBinaryOrbitPriors['P_days']/365.25), sd=0.7, 
                                      testval=np.log(InitBinaryOrbitPriors['P_days']/365.25)) # years
                loga0 = pm.Normal("loga0", mu=np.log(InitBinaryOrbitPriors["a0"]), sd=2.0,
                              testval=np.log(InitBinaryOrbitPriors["a0"]))
                cosi = pm.Uniform('cosi', lower=-1., upper=1., testval=InitBinaryOrbitPriors['cosi']) 
                se = np.sqrt(InitBinaryOrbitPriors["e"]) if InitBinaryOrbitPriors["e"] > 0 else 0.05
                ecs = pmx.UnitDisk("ecs", testval=np.array(
                    [se * np.cos(InitBinaryOrbitPriors["omega"]), se * np.sin(InitBinaryOrbitPriors["omega"])]))
                Omega = pmx.Angle("Omega", testval=InitBinaryOrbitPriors["Omega"])
                phase = pmx.Angle("phase", testval=InitBinaryOrbitPriors["phase"])
        
            
            period_years = pm.Deterministic('period_years', tt.exp(logPeriod))
            period = pm.Deterministic('period', period_years * 365.25)
            a0_mas = pm.Deterministic("a0_mas", tt.exp(loga0))                    # mas  
            # a0_au = pm.Deterministic("a0_au", a0_mas / Parallax)
            incl = pm.Deterministic('incl', tt.arccos(cosi))
            pm.Deterministic("incl_deg", incl * 180.0 / np.pi)
        
            ecc = pm.Deterministic("ecc", tt.sum(ecs ** 2))
            omega = pm.Deterministic("omega", tt.arctan2(ecs[1], ecs[0]))
            pm.Deterministic("omega_deg", omega * 180.0 / np.pi) # Argument of periastron
            pm.Deterministic("Omega_deg", Omega * 180.0 / np.pi) # Longitude of Ascending Node
            tperi = pm.Deterministic("tperi", period_years * phase / (2.0 * np.pi))  # yr, J2017.5
            
            cw, sw = tt.cos(omega), tt.sin(omega)
            cO, sO = tt.cos(Omega), tt.sin(Omega)
            A = pm.Deterministic("A_TI", a0_mas * (cw * cO - sw * sO * cosi))
            B = pm.Deterministic("B_TI", a0_mas * (cw * sO + sw * cO * cosi))
            F = pm.Deterministic("F_TI", -a0_mas * (sw * cO + cw * sO * cosi))
            G = pm.Deterministic("G_TI", -a0_mas * (sw * sO - cw * cO * cosi))    
        
            # Define the orbit
            orbit = xo.orbits.KeplerianOrbit(period=period_years, a=-a0_mas,
                                             t_periastron=tperi, 
                                             incl=incl, Omega=Omega,
                                             ecc=ecc, omega=omega, )
                                             # m_star=mstar, r_star=rstar,
                                             # m_planet=mp)

            t0 = pm.Deterministic("t0", orbit.t0*365.25 + JD_REF_DR4)
        
            x_orb, y_orb, _ = orbit.get_planet_position(AstroDataset['dt'].values)
            # x = North (Dec offset)
            # y = East (RA offset)
        
            w_orb = pm.Deterministic("w_orb", y_orb * tt.sin(Theta) + x_orb * tt.cos(Theta))
            w_BSModel = pm.Deterministic("w_BSModel", w_SSModel+w_orb)
            Residuals = AstroDataset['w'].values - w_BSModel

            #### Things needed for plotting (binary) ####
            RA_model_BS = pm.Deterministic("RA_model_BS", RA_model_SS + y_orb)
            Dec_model_BS = pm.Deterministic("Dec_model_BS", Dec_model_SS + x_orb)
            RA_data_BS = pm.Deterministic("RA_data_BS", RA_model_BS + Residuals * tt.sin(Theta))
            Dec_data_BS = pm.Deterministic("Dec_data_BS", Dec_model_BS + Residuals * tt.cos(Theta))

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
                # relative to X_RVs_Ref (median RV time, in BJD)
                # m/s/year & m/s/year2

                if LinRVTrend: RVTrend = pm.Normal("RVTrend", mu=0.0, sd=100.0, shape=1, testval=0.0)             
                else: RVTrend = pm.Normal("RVTrend", mu=0.0, sd=100.0, shape=2, testval=np.array([0.0, 0.0])) 
                
                # Per instrument parameters
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
                    
                    vrad = - orbit.get_radial_velocity(t, K=K_RV)
                    pm.Deterministic("vrad" + name, vrad)

                    if LinRVTrend: bkg = pm.Deterministic("bkg" + name, RVTrend[0]*(t-t_ref))
                    else: bkg = pm.Deterministic("bkg" + name, RVTrend[0]*(t-t_ref)  + RVTrend[1]*((t-t_ref)**2))
                    
                    return pm.Deterministic("rv_model" + name, vrad + bkg)
                
                # Define the RVs at the observed times
                rv_model = get_rv_model((X_RVs - JD_REF_DR4)/365.25)
                
                # Also define the model on a fine grid as computed above (for plotting)
                rv_model_pred = get_rv_model((t_rv - JD_REF_DR4)/365.25, name="_pred")
                
                rv_loglike = pm.Normal("rv_obs", mu=rv_model, sd=tt.sqrt(RVDiag), observed=OffsetSubtracted)
            
            ############################################
            ############################################

        
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
        print("====\nStarting MAP optimization\n====")        
        map_soln = model.test_point
        map_soln = pmx.optimize(map_soln, vars=[Parallax,  PMRA, PMDec, dRA, dDec, logAstroJitter])
        map_soln = pmx.optimize(map_soln)   # final joint stage, everything moves together
        print("====\nFinished MAP optimization\n====")
            
print(model.logp(map_soln))
for k, v in sorted(map_soln.items()):
    if not k.endswith("__") and np.ndim(v) == 0:
        print("  %-14s % .6g" % (k, float(v)))

########################################################################
########################################################################

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
var_names = ['PMRA', 'PMDec', 'dRA', 'dDec', 'logAstroJitter', 'Parallax']

if FitBinary:
    var_names.append("logPeriod")
    var_names.append("period")
    var_names.append("period_years")    
    var_names.append("loga0")    
    var_names.append("a0_mas")        
    # var_names.append("a0_au")            
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

TracesSummary, ax = plt.subplots(figsize=(12,8))
ax.axis('tight')
ax.axis('off')
ax.set_title('Summary from {} chains'.format(trace.nchains))
the_table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
the_table.auto_set_font_size(False)
# plt.title(ComparisonText, size=15)
plt.tight_layout()
# TracesSummary.savefig(os.path.join(ResultDirectory, 'TraceSummary.png'))
# df.to_csv(os.path.join(ResultDirectory, 'ChainSummary_{}.csv'.format(RunName)))



with model:
	TracePlot = pm.traceplot(trace, var_names=var_names)
	TraceFigure = TracePlot[0][0].get_figure()
	# TraceFigure.savefig(os.path.join(ResultDirectory, 'TraceFigure.png'))
	samples = pm.trace_to_dataframe(trace, varnames=var_names)


print("==============\nFinished Generating Trace Results\n==============")
########## Corner Plot #################
try:
	CornerPlot = corner.corner(samples, quantiles=[0.16, 0.5, 0.84],
			show_titles=True, title_kwargs={"fontsize": 12}, use_math_text=True)
except:
	MCMCColumns = np.array(samples.columns)
	iii = [~ np.any(np.isnan(samples[c])) for c in MCMCColumns]
	NonNanColumns =  MCMCColumns[iii]
	print("NaN columns = "+ MCMCColumns[~np.array(iii)])
	CornerPlot = corner.corner(samples[NonNanColumns], quantiles=[0.16, 0.5, 0.84],
			show_titles=True, title_kwargs={"fontsize": 12}, use_math_text=True)
print("==============\nFinished Generating Corner Plot \n==============")

Parameters = np.array([[c, *np.percentile(samples[c], [50, 16, 84])] for c in samples.columns])

Median = Parameters[:,1].astype(float)
LSigma = Parameters[:,2].astype(float) - Parameters[:,1].astype(float)
USigma = Parameters[:,3].astype(float) - Parameters[:,1].astype(float)
# ParameterResults = pd.DataFrame('Parameters':Parameters)
LatexString = []

for i in range(len(LSigma)):
	if np.isclose(LSigma[i], USigma[i], rtol=1e-5):
		LatexString.append(r'{:.7f}$\pm${:.7f}'.format(Median[i], USigma[i]))
	else:
		LatexString.append(r'{:.7f}$^{{+{:.7f}}}_{{-{:.7f}}}$'.format(Median[i], USigma[i], np.abs(LSigma[i])))

ParameterDF = pd.DataFrame({'Parameters':Parameters[:,0], 'Value':np.array(LatexString)}) #,
# ParameterDF.to_csv(os.path.join(ResultDirectory, 'ParameterSummary_{}.dat'.format(RunName)), index=False)


########################################################################
########################################################################

if FitBinary & FitRV:
    MCMCPosterior = PlotRVCurve(RVDict=RVDict, RV_GP=False,
    		trace=trace, MakePlots=False, Title='MCMC', NPlanets=1)
    

########################################################################
########################################################################
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
# pp.savefig(CornerPlot)
# pp.savefig(ParameterSummary)

if FitBinary & FitRV: 
    pp.savefig(MCMCPosterior)

pp.close()
print("==============\nSaved everything for {}\n==============".format(RunName))
########################################################################
########################################################################



