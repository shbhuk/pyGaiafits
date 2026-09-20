import sys
import os
from xo_utils import BinByEpoch, GetModel
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


PlanetNames = ['b', 'c', 'd', 'e', 'f']
PlanetColours = ["C0", "C1", "C2", "C3", "C4", "C5", "Magenta"]
DarkerInstColours = ["darkgreen", "darkslateblue", "darkred",
                     "darkviolet", "darksalmon", "orange", "palevioletred"]

try:
    pwd = os.path.dirname(os.path.abspath(__file__))
except:
    pwd = r'/home/skanodia/work/pyGaiafits/Code'


print(pwd)
sys.path.append(pwd)


def PlotRVCurve(RVDict, RV_GP=False,
                soln=None, trace=None, MakePlots=False, Title=None, NPlanets=1):
    """
    soln : Output of Model MAP optimization.
    trace: Output of MCMC chains.
    Define one of those two.

    """

    X_RVs = RVDict['X_RVs']
    Y_RVs = RVDict['Y_RVs']
    Yerr_RVs = RVDict['Yerr_RVs']
    t_rv = RVDict['t_rv']
    RVInstruments = RVDict['RVInstruments']
    InstID = RVDict['InstID']

    if soln is not None:
        Offset = soln['RVMean']
        RVDiag = soln['RVDiag']
        if RV_GP:
            GP = soln['rv_gp']
            GP_pred = soln['rv_gp_pred']
        else:
            GP = np.zeros(len(Offset))
            GP_pred = np.zeros(len(t_rv))

        rv_model_pred = soln["rv_model_pred"]
        rv_model = soln["rv_model"]
        vrad_pred = soln["vrad_pred"]

        ResultType = 'MAP'
    else:
        Offset = np.median(trace.RVMean, axis=0)
        RVDiag = np.median(trace.RVDiag, axis=0)
        if RV_GP:
            GP = np.median(trace.rv_gp, axis=0)
            GP_pred = np.median(trace.rv_gp_pred, axis=0)
        else:
            GP = np.zeros(len(Offset))
            GP_pred = np.zeros(len(t_rv))

        rv_model_pred = np.percentile(
            trace.rv_model_pred, q=[16, 50, 84], axis=0)
        rv_model = np.median(trace.rv_model, axis=0)
        vrad_pred = np.median(trace.vrad_pred, axis=0)

        ResultType = 'MCMC'

    # Subtract the instrument specific offsets here
    OffsetSubtractedRVs = Y_RVs - Offset

    DetrendedRVs = OffsetSubtractedRVs - GP
    Residuals = OffsetSubtractedRVs - rv_model

    fig, axes = plt.subplots(3, 1, figsize=(10, 5), sharex=True)
    fig.subplots_adjust(right=0.75)

    ax = axes[0]

    for i, RVInst in enumerate(RVInstruments):
        m = InstID == i
        ax.errorbar(
            X_RVs[m], OffsetSubtractedRVs[m], yerr=Yerr_RVs[m], fmt="o", label=RVInst, c=DarkerInstColours[i], ms=3,

        )

    ax.plot(t_rv, GP_pred, c='k', label="GP")
    ax.axhline(0, color="k", lw=1)
    ax.legend(fontsize=8, loc='upper left',
              bbox_to_anchor=(1.01, 1), borderaxespad=0)
    # ax.set_xlabel("BJD")
    # ax.set_ylabel("RVs [m/s]")
    # ax.set_title("Offset Subtracted RVs")

    ax = axes[1]
    ax.errorbar(X_RVs, DetrendedRVs, yerr=Yerr_RVs,
                fmt=".k", label='Detrended RVs')

    if NPlanets > 1:
        for p in range(NPlanets):
            ax.plot(t_rv, vrad_pred[:, p], "--", alpha=0.5,
                    label='Model Planet '+PlanetNames[p], c=PlanetColours[p])

    else:
        ax.plot(t_rv, vrad_pred, "--k", alpha=0.5, label='Model Planet b')

    if ResultType == 'MCMC':
        ax.plot(t_rv, rv_model_pred[1], label="Combined", c="darkslategrey")
        ax.fill_between(
            t_rv, rv_model_pred[0], rv_model_pred[2], alpha=0.3, color="darkslategrey")
    else:
        ax.plot(t_rv, rv_model_pred, label="Combined", c="darkslategrey")
    ax.legend(fontsize=8, loc='upper left',
              bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.set_ylabel("RV [m/s]")
    ax.axhline(0, color="k", lw=1)

    ax = axes[2]

    for i, RVInst in enumerate(RVInstruments):
        m = InstID == i
        ax.errorbar(
            X_RVs[m], Residuals[m]/Yerr_RVs[m], yerr=np.sqrt(RVDiag[m])/Yerr_RVs[m], fmt="o", c=DarkerInstColours[i], label=RVInst + ' $\sigma$ + Jitter = {:.2f} m/s'.format(np.median(np.sqrt(RVDiag[m])))
        )
        ax.errorbar(
            X_RVs[m], Residuals[m]/Yerr_RVs[m], yerr=Yerr_RVs[m]/Yerr_RVs[m], fmt=".k", label=RVInst + ' $\sigma$ = {:.2f} m/s'.format(np.median(Yerr_RVs[m])), zorder=100
        )

    # err = np.sqrt(yerr_rv**2 + jitter_RV**2)
    # ax.errorbar(x_rv, Residuals, yerr=err, fmt=".r", label='HPF + Jitter = {:.2f} m/s'.format(np.median(err)))
    # ax.errorbar(x_rv, Residuals, yerr=yerr_rv, fmt=".k", label='HPF = {:.2f} m/s'.format(np.median(yerr_rv)))
    ax.axhline(0, color="k", lw=1)
    ax.legend(fontsize=8, loc='upper left',
              bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.set_ylabel("Residuals/$\sigma$")
    ax.set_xlim(t_rv.min(), t_rv.max())
    ax.set_xlabel("BJD_TDB")

    if Title is None:
        if ResultType == 'MAP':
            """
            Title = \
            " MAP Model\nP={:.3f} d, $\omega =$ {:.2f}$^\circ$, ecc = {:.2f}\n \
            $\gamma =$ {:.2f} m/s,  dv/dt = {:.2f} m/s/day,  Jitter = {:.2f} m/s\n \
            M$_p$ = {:.2f} M$_{{\oplus}}$,  M$_*$ = {:.3f} M$_{{\odot}}$".format(
            soln['period'],soln['omega']*180/np.pi, soln['ecc'],
            soln['rv0'], soln['rvtrend'],  soln['jitter_RV'],
            soln['m_pl'], soln['m_star'])
            """
            Title = "MAP run"
        else:
            Title = 'MCMC run'

    _ = axes[0].set_title(Title, size=15)
    # plt.tight_layout()
    plt.grid(False)

    if MakePlots:
        plt.show(block=False)

    fig.subplots_adjust(hspace=0.1)

    return fig

def ResidualPlots(xdata, ydata,
                  xmodel, ymodel, ymodel_sigma=None,
                  Title='', Xlabel='', Ylabel='', Ymodellabel='',
                  ResidScale=1e6, ResidUnits='ppm'):
    """
    Make residual plot.
    ydata can be a list of 1D arrays, in which case will plot each one separately.
    In such case, give names for each dataset in Ydatalabels
    ymodel_sigma : Default is None. If specified then will shade the 1 sigma uncertainties of the model
 
    ResidScale / ResidUnits : scaling and label for the residual sigma. The
    1e6 / 'ppm' default is right for relative flux; pass 1.0 / 'mas' for
    astrometry.
    """
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].scatter(xdata, ydata, c='k', label='Data', s=3, alpha=0.3)
    axes[0].plot(xmodel, ymodel, color='b', label=Ymodellabel,
                 markersize=3, alpha=0.7, rasterized=True)
    R = np.nanstd(ydata - ymodel)*ResidScale
    axes[1].scatter(xdata, ydata - ymodel, c='k',
                    label="Sigma = {:.2f} {}".format(R, ResidUnits), s=3, alpha=0.3, rasterized=True)
    if ymodel_sigma is not None:
        art = axes[0].fill_between(
            xmodel, ymodel_sigma[0], ymodel_sigma[1], color='b', alpha=0.4)
        art.set_edgecolor("none")
    axes[1].set_xlabel(Xlabel)
    axes[0].set_ylabel(Ylabel)
    axes[1].set_ylabel('Residuals')
    axes[0].set_title(Title)
    axes[0].legend()
    axes[1].legend()
    fig.subplots_adjust(hspace=0.02)
    plt.tight_layout()
    return fig, axes

def PlotAstroAL(data, soln=None, trace=None, outdir=None, label='',
                ModelKey='w_SSModel', OrbitKey=None, EpochKey='transit_id',
                SavePlot=False, MakePlots=False, Title=None):
    """
    soln : Output of Model MAP optimization.
    trace: Output of MCMC chains.
    Define one of those two.
 
    Built on ResidualPlots, with the inverse-variance weighted mean of each
    epoch overplotted on both panels -- the same pattern as the 300 s bins in
    PlotPhaseFoldedLightCurve.
 
    ModelKey : 'w_SSModel' or 'w_BSModel'.
    OrbitKey : 'w_orb'. If given, the top panel shows the orbit signal with
               the 5p model removed rather than the raw AL abscissa.
    """
    ResultType = 'MAP' if soln is not None else 'MCMC'
    w_med, w_lo, w_hi = GetModel(ModelKey, soln, trace)
 
    t = np.asarray(data.t_jyear, dtype=float)
    wObs = np.asarray(data.w, dtype=float)
    wErr = np.asarray(data.w_err, dtype=float)
    Order = np.argsort(t)
 
    if OrbitKey is not None:
        o_med, o_lo, o_hi = GetModel(OrbitKey, soln, trace)
        YData = wObs - (w_med - o_med)
        YModel = o_med
        YSigma = None if o_lo is None else [o_lo[Order], o_hi[Order]]
        Ylabel = "orbit AL [mas]"
        Ymodellabel = "{} orbit".format(ResultType)
    else:
        YData = wObs
        YModel = w_med
        YSigma = None if w_lo is None else [w_lo[Order], w_hi[Order]]
        Ylabel = "AL centroid [mas]"
        Ymodellabel = "{} model".format(ResultType)
 
    if Title is None:
        Title = "{} {}".format(label, ResultType)
 
    fig, axes = ResidualPlots(xdata=t, ydata=YData,
                              xmodel=t[Order], ymodel=YModel[Order],
                              ymodel_sigma=YSigma,
                              Title=Title, Xlabel="Time [Julian year]",
                              Ylabel=Ylabel, Ymodellabel=Ymodellabel,
                              ResidScale=1.0, ResidUnits='mas')
 
    # epoch weighted means over the top, on both panels
    DoBin = EpochKey is not None and EpochKey in getattr(data, 'columns', [])
    if DoBin:
        EpochID = np.asarray(data[EpochKey])
        tB, yB, eB = BinByEpoch(t, YData, wErr, EpochID)
        axes[0].errorbar(tB, yB, yerr=eB, fmt='o', ms=4, color='r', capsize=2,
                         lw=1.2, zorder=5,
                         label='{} epochs, weighted mean'.format(len(tB)))
        tB, rB, eB = BinByEpoch(t, wObs - w_med, wErr, EpochID)
        axes[1].errorbar(tB, rB, yerr=eB, fmt='o', ms=4, color='r', capsize=2,
                         lw=1.2, zorder=5,
                         label='epoch mean, RMS {:.3f} mas'.format(np.nanstd(rB)))
        axes[0].legend(fontsize=8)
        axes[1].legend(fontsize=8)
 
    axes[1].axhline(0, color='0.5', lw=0.8, zorder=-5)
 
    if SavePlot and outdir is not None:
        fig.savefig(os.path.join(outdir, 'AstroAL_{}_{}.png'.format(label, ResultType)), dpi=300)
    if MakePlots:
        plt.show(block=False)
    return fig
 
 
########################################################################
# 2. Sky motion, astrometric orbit and both residual series
########################################################################

 
 
def BinSkyByEpoch(t, RA, Dec, wErr, EpochID):
    """
    Epoch-average a reconstructed on-sky position.
 
    The reconstruction is  model + residual*sin(Theta) / model + residual*cos(Theta),
    and Theta is fixed within an epoch, so all visits of one epoch lie along a
    single line -- the streaks you see if these are plotted unbinned. Because
    sin/cos(Theta) are constant across the epoch, inverse-variance averaging
    the coordinate is identical to averaging the AL residual and then
    projecting it, so BinByEpoch can be applied to each coordinate directly.
    """
    _, RAb, _ = BinByEpoch(t, RA, wErr, EpochID)
    _, Decb, _ = BinByEpoch(t, Dec, wErr, EpochID)
    return RAb, Decb


def PlotAstroSky(data, soln=None, trace=None, outdir=None, label='',
                 EpochKey='transit_id', NDraws=40, SavePlot=False,
                 MakePlots=False, Title=None):
    """
    Four panels: motion on the sky, single-star AL residuals, astrometric
    orbit, binary-star AL residuals. Drops to 1x2 when w_BSModel is absent.
 
    ResidualPlots is not used here -- it builds its own two-panel figure, so
    it cannot draw into this grid. These residual panels are plain black
    points with the epoch weighted means over the top.
    """
    ResultType = 'MAP' if soln is not None else 'MCMC'
    Keys = soln.keys() if soln is not None else trace.varnames
    HasBinary = 'w_BSModel' in Keys
 
    DoBin = EpochKey is not None and EpochKey in getattr(data, 'columns', [])
    EpochID = np.asarray(data[EpochKey]) if DoBin else None
    VisitAlpha = 0.18 if DoBin else 0.5
 
    t = np.asarray(data.t_jyear, dtype=float)
    wObs = np.asarray(data.w, dtype=float)
    wErr = np.asarray(data.w_err, dtype=float)
    Order = np.argsort(t)
 
    def ResidPanel(Ax, Resid, PanelTitle):
        Chi2 = np.nansum((Resid / wErr) ** 2)
        NData = int(np.sum(np.isfinite(Resid)))
        Ax.errorbar(t, Resid, yerr=wErr, fmt='.k', ms=3, alpha=VisitAlpha,
                    label="RMS {:.2f} mas, chi2/N = {:.2f}".format(
                        np.nanstd(Resid), Chi2 / max(NData, 1)))
        if DoBin:
            tB, yB, eB = BinByEpoch(t, Resid, wErr, EpochID)
            Ax.errorbar(tB, yB, yerr=eB, fmt='o', ms=4, color='k', capsize=2,
                        lw=1.2, zorder=5,
                        label="epoch mean, RMS {:.3f} mas".format(np.nanstd(yB)))
        Ax.axhline(0, color='0.5', lw=0.8)
        Ax.set_xlabel("Time [Julian year]")
        Ax.set_ylabel("Residual [mas]")
        Ax.set_title(PanelTitle)
        Ax.legend(loc='best', fontsize=8)
 
    RA_SS, _, _ = GetModel('RA_model_SS', soln, trace)
    Dec_SS, _, _ = GetModel('Dec_model_SS', soln, trace)
    w_SS, _, _ = GetModel('w_SSModel', soln, trace)
 
    if HasBinary:
        RA_BS, _, _ = GetModel('RA_model_BS', soln, trace)
        Dec_BS, _, _ = GetModel('Dec_model_BS', soln, trace)
        RAd, _, _ = GetModel('RA_data_BS', soln, trace)
        Decd, _, _ = GetModel('Dec_data_BS', soln, trace)
        RAg, _, _ = GetModel('RA_orbit_grid', soln, trace)
        Decg, _, _ = GetModel('Dec_orbit_grid', soln, trace)
        w_BS, _, _ = GetModel('w_BSModel', soln, trace)
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    else:
        RAd, _, _ = GetModel('RA_data_SS', soln, trace)
        Decd, _, _ = GetModel('Dec_data_SS', soln, trace)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes = np.array([axes])
 
    Ax = axes[0, 0]
    Ax.plot(RAd, Decd, '.k', ms=3, alpha=VisitAlpha, label="Reconstructed observations")
    if DoBin:
        RAb, Decb = BinSkyByEpoch(t, RAd, Decd, wErr, EpochID)
        Ax.plot(RAb, Decb, 'ok', ms=4, zorder=5,
                label="{} epochs, weighted mean".format(len(RAb)))
    Ax.plot(RA_SS[Order], Dec_SS[Order], '-r', lw=1.2, label="Single Star Model")
    if HasBinary:
        Ax.plot(RA_BS[Order], Dec_BS[Order], '-b', lw=1.2, label="Binary Star Model")
    Ax.set_xlabel(r"$\Delta\alpha\cos(\delta)$ [mas]")
    Ax.set_ylabel(r"$\Delta\delta$ [mas]")
    Ax.set_title("Motion on the sky")
    Ax.invert_xaxis()
    Ax.legend(loc='best', fontsize=8)
 
    ResidPanel(axes[0, 1], wObs - w_SS, "Single star model AL residuals")
 
    if HasBinary:
        Ax = axes[1, 0]
        Ax.plot(RAd - RA_SS, Decd - Dec_SS, '.k', ms=3, alpha=VisitAlpha,
                label="Reconstructed observations")
        if DoBin:
            RAb, Decb = BinSkyByEpoch(t, RAd - RA_SS, Decd - Dec_SS, wErr, EpochID)
            Ax.plot(RAb, Decb, 'ok', ms=4, zorder=5,
                    label="{} epochs, weighted mean".format(len(RAb)))
        if trace is not None and NDraws > 0:
            Idx = np.random.choice(len(trace['RA_orbit_grid']),
                                   size=min(NDraws, len(trace['RA_orbit_grid'])),
                                   replace=False)
            for i in Idx:
                Ax.plot(trace['RA_orbit_grid'][i, :-2], trace['Dec_orbit_grid'][i, :-2],
                        '-b', lw=0.6, alpha=0.12, zorder=-5)
        Ax.plot(RAg[:-2], Decg[:-2], '-b', lw=1.6, label="{} orbit".format(ResultType))
        Ax.plot(0, 0, '+k', ms=9)
        Ax.set_xlabel(r"$\Delta\alpha\cos(\delta)$ [mas]")
        Ax.set_ylabel(r"$\Delta\delta$ [mas]")
        Ax.set_title("Astrometric orbit")
        Ax.invert_xaxis()
        Ax.legend(loc='best', fontsize=8)
 
        ResidPanel(axes[1, 1], wObs - w_BS, "Binary star model AL residuals")
 
    if Title is None:
        Title = "{} Fit for {}".format(ResultType, label)
    fig.suptitle(Title)
    fig.tight_layout()
    if SavePlot and outdir is not None:
        fig.savefig(os.path.join(outdir, 'AstroSky_{}_{}.png'.format(label, ResultType)), dpi=300)
    if MakePlots:
        plt.show(block=False)
    return fig

