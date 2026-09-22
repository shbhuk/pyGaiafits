"""
Figures for the Gaia DR4 epoch-astrometry fits driven by ``xo_Astrometry.py``.

Every function here takes a fit result -- either a MAP solution dict or a
PyMC3 trace -- together with the data it was fit to, and returns a
matplotlib Figure. Nothing is written to disk unless ``SavePlot=True`` is
passed; ``xo_Astrometry.py`` instead collects the returned figures into a
single multi-page PDF.

    ``PlotAstroAL``    along-scan abscissa, or the isolated orbit signal,
                       against time with residuals underneath
    ``PlotAstroSky``   reconstructed motion on the sky, the astrometric
                       orbit, and the residuals of both models
    ``PlotRVCurve``    three-panel radial-velocity figure, used only by the
                       ``Binary+RV`` analysis
    ``ResidualPlots``  the generic data-over-model / residual pair that
                       ``PlotAstroAL`` is built on
    ``BinSkyByEpoch``  epoch-averaging helper for the sky panels

MAP and MCMC
------------
Each function accepts ``soln`` or ``trace`` and works out which it was
given; ``soln`` wins if both are passed. Model values are pulled through
``GetModel`` from ``xo_utils``, which returns a median plus a 16th/84th
percentile band for a trace, and a bare value with no band for a MAP
solution. That is why shaded uncertainty bands appear only on the MCMC
versions of these figures.

Epoch binning
-------------
Gaia records several CCD observations per field-of-view transit. Plotted
raw, they show up as short streaks: within a transit the scan angle is
fixed and only the along-scan coordinate is measured, so all of a transit's
visits fall on one line. The astrometry figures therefore overplot the
inverse-variance weighted mean of each transit, keyed on the ``transit_id``
column, through ``BinByEpoch``.

Backend
-------
``matplotlib.use("Agg")`` is set at import, before any figure is created.
That makes the module safe to run headless on a cluster, but it also means
the ``MakePlots=True`` arguments below cannot display anything --
``plt.show`` is a no-op under Agg. Work with the returned Figure instead.

Figures are never closed here, so a long loop over many sources will
accumulate them; call ``plt.close(fig)`` once a figure has been saved.
"""
import sys
import os
from xo_utils import BinByEpoch, GetModel
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


# Per-planet labels and colours, used only by PlotRVCurve's multi-planet
# branch. DarkerInstColours is indexed by RV instrument number, so it
# caps the number of instruments that can be drawn at seven.
PlanetNames = ['b', 'c', 'd', 'e', 'f']
PlanetColours = ["C0", "C1", "C2", "C3", "C4", "C5", "Magenta"]
DarkerInstColours = ["darkgreen", "darkslateblue", "darkred",
                     "darkviolet", "darksalmon", "orange", "palevioletred"]

# Kept for consistency with how these modules are launched elsewhere.
# Note it has no effect on this module's own imports: the
# `from xo_utils import ...` line above is resolved before this runs.
try:
    pwd = os.path.dirname(os.path.abspath(__file__))
except:
    pwd = r'/home/skanodia/resgroupdir/pyGaiafits/Code'


print(pwd)
sys.path.append(pwd)


def PlotRVCurve(RVDict, RV_GP=False,
                soln=None, trace=None, MakePlots=False, Title=None, NPlanets=1):
    """
    Three-panel radial-velocity figure for the Binary+RV analysis.

    soln : Output of Model MAP optimization.
    trace: Output of MCMC chains.
    Define one of those two.

    Panels, top to bottom:
      1. RVs with the per-instrument offsets removed, coloured by
         instrument, over the GP prediction.
      2. The same RVs detrended, over the Keplerian model. For an MCMC
         result the combined model is a median with a 16th-84th
         percentile band; for a MAP result it is a single curve.
      3. Residuals divided by the formal uncertainty, drawn twice per
         instrument: once with error bars widened by the fitted jitter,
         once with the formal errors alone.

    Parameters
    ----------
    RVDict : dict
        The RV bundle assembled by xo_Astrometry.py. The keys read here
        are X_RVs (BJD), Y_RVs and Yerr_RVs (m/s), t_rv (the fine
        prediction grid), RVInstruments (labels, in the order their
        offsets were fit) and InstID (index into that list, one entry
        per measurement). The per-instrument x_rv_*/y_rv_*/yerr_rv_*
        entries of RVDict are not used.
    RV_GP : bool, optional
        Whether the fit included a GP on the RVs. When False, zeros
        stand in for it -- so panel 1's line labelled 'GP' is a flat
        zero rather than a fitted component. It is drawn either way.
    soln, trace : dict or pymc3.MultiTrace, optional
        Give exactly one. Either must supply RVMean, RVDiag, rv_model,
        rv_model_pred and vrad_pred, plus rv_gp and rv_gp_pred when
        RV_GP is True. RVDiag is a variance (formal error squared plus
        jitter squared), which is why panel 3 takes its square root.
    MakePlots : bool, optional
        Calls plt.show, which does nothing under the Agg backend forced
        at import. See the module docstring.
    Title : str, optional
        Title for the top panel; defaults to 'MAP run' or 'MCMC run'.
    NPlanets : int, optional
        Number of Keplerians in the model. Above 1, vrad_pred is indexed
        per planet in panel 2. The astrometry pipeline always passes 1.

    Returns
    -------
    matplotlib.figure.Figure
    """

    X_RVs = RVDict['X_RVs']
    Y_RVs = RVDict['Y_RVs']
    Yerr_RVs = RVDict['Yerr_RVs']
    t_rv = RVDict['t_rv']
    RVInstruments = RVDict['RVInstruments']
    InstID = RVDict['InstID']

    # MAP and MCMC differ only in how the model values are extracted: a
    # MAP solution is read straight out of the dict, while a trace is
    # reduced to a posterior median (and, for the combined model, a
    # 16/50/84 percentile triple used to shade the band in panel 2).
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

    # Panels share an x axis; the right margin is pulled in to leave room
    # for the legends, which are anchored outside the axes.
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

    # The triple-quoted block below is a disabled title template, not a
    # docstring: it is a bare string expression that Python evaluates and
    # discards. It refers to keys (m_pl, m_star, rv0, jitter_RV) that the
    # astrometric model does not define, so it cannot be re-enabled as-is.
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
    Generic two-panel figure: data with a model over it, residuals below.

    Parameters
    ----------
    xdata, ydata : array_like
        The observations, drawn as a scatter in the upper panel.
    xmodel, ymodel : array_like
        The model curve. Note these serve two purposes that pull in
        different directions: ymodel is drawn against xmodel, but the
        residuals are formed element-wise as ydata - ymodel. So ymodel
        must be aligned with ydata index-for-index, which means passing
        a sorted model against sorted xmodel is only correct when xdata
        was already sorted. See the note in PlotAstroAL.
    ymodel_sigma : sequence of two arrays, optional
        Lower and upper bounds shaded around the model curve. Ignored
        when None, which is what a MAP solution yields.
    Title, Xlabel, Ylabel, Ymodellabel : str, optional
        Text for the title, the shared x axis, the upper y axis, and the
        model's legend entry.
    ResidScale, ResidUnits : float and str, optional
        Scale factor and label for the residual scatter quoted in the
        lower panel's legend. The 1e6 / 'ppm' defaults suit relative
        flux; the astrometry callers pass 1.0 / 'mas'.

    Returns
    -------
    fig, axes
        The Figure and its two Axes, so a caller can draw more onto
        them -- which is how PlotAstroAL adds its epoch means.
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
    epoch overplotted on both panels.
 
    ModelKey : 'w_SSModel' or 'w_BSModel'.
    OrbitKey : 'w_orb'. If given, the top panel shows the orbit signal with
               the 5p model removed rather than the raw AL abscissa.

    Parameters
    ----------
    data : pandas.DataFrame
        Epoch astrometry from load_epoch_astrometry. The columns t_jyear,
        w, w_err and (for binning) transit_id are read.
    soln, trace : dict or pymc3.MultiTrace, optional
        Give exactly one. Must contain ModelKey, and OrbitKey too when
        that is requested.
    outdir : str, optional
        Directory for the png, used only when SavePlot is True.
    label : str, optional
        Free-text tag put in the title and the output filename, e.g.
        'Single Star MAP'.
    EpochKey : str, optional
        Column grouping observations into epochs, default 'transit_id'.
        Binning is skipped silently if the column is absent or this is
        None.
    SavePlot : bool, optional
        Also write AstroAL_<label>_<MAP|MCMC>.png into outdir.
    MakePlots : bool, optional
        No-op under the Agg backend; see the module docstring.
    Title : str, optional
        Overrides the '<label> <MAP|MCMC>' default.

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    The residual panel assumes `data` arrives sorted in time. The model
    handed to ResidualPlots is sorted by `Order`, while the observations
    are not, and ResidualPlots forms residuals element-wise -- so the two
    only line up when `Order` is the identity. xo_Astrometry.py sorts the
    table on 'dt' before fitting, which makes that true there. Passing an
    unsorted table would leave the top panel correct but silently scramble
    the residual panel and its quoted scatter. The epoch-binned residuals
    added further down are computed separately and are unaffected.
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
 
    # Epoch weighted means over the top, on both panels. The upper panel
    # bins whatever is being displayed there (the abscissa, or the
    # isolated orbit signal); the lower panel always bins the full
    # residual wObs - w_med, regardless of OrbitKey.
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

    Parameters
    ----------
    t : array_like
        Observation times, passed through to BinByEpoch for grouping.
    RA, Dec : array_like
        Reconstructed offsets in mas, one entry per observation.
    wErr : array_like
        Along-scan uncertainties, used as the weights for both
        coordinates.
    EpochID : array_like
        Epoch label per observation, normally transit_id.

    Returns
    -------
    RAb, Decb : ndarray
        One entry per epoch, ordered by time. The binned times
        themselves are discarded, since the sky panels plot Dec against
        RA rather than either against time.
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

    The layout is chosen from the result itself: if 'w_BSModel' is among
    the model's variables the figure is 2x2, otherwise 1x2.

    Parameters
    ----------
    data : pandas.DataFrame
        Epoch astrometry from load_epoch_astrometry; t_jyear, w, w_err
        and transit_id are read.
    soln, trace : dict or pymc3.MultiTrace, optional
        Give exactly one. Both cases need RA_model_SS, Dec_model_SS and
        w_SSModel. A single-star result additionally needs RA_data_SS
        and Dec_data_SS; a binary result needs RA_model_BS,
        Dec_model_BS, RA_data_BS, Dec_data_BS, RA_orbit_grid,
        Dec_orbit_grid and w_BSModel.
    outdir, label, SavePlot, MakePlots, Title
        As for PlotAstroAL; the png is named AstroSky_<label>_<type>.png.
    EpochKey : str, optional
        Column grouping observations into epochs, default 'transit_id'.
        When binning is off, the unbinned points are drawn at a higher
        alpha to stay legible on their own.
    NDraws : int, optional
        Number of posterior orbits drawn faintly behind the median one,
        in the orbit panel. Applies to a trace only. The draws are chosen
        with np.random.choice and are not seeded, so the faint curves
        differ between runs on the same trace.

    Returns
    -------
    matplotlib.figure.Figure
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
 
    # Residual panel shared by the single-star and binary rows. The chi2
    # in the legend is divided by the number of finite points, not by a
    # degrees-of-freedom count, and treats every CCD observation as
    # independent -- so read it as a scatter diagnostic, not a goodness
    # of fit.
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
 
    # Only the medians are needed here: the sky panels draw curves and
    # points rather than shaded bands, so the percentile edges that
    # GetModel returns for a trace are discarded.
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
        # The orbit grid spans a full 2*pi inclusive of both endpoints, so
        # its last points retrace the first; dropping the final two keeps
        # the curve from doubling back over itself.
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

