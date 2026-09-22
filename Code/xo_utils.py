# -*- coding: utf-8 -*-
"""
Support library for the Gaia DR4 pre-release epoch-astrometry fits driven by
``xo_Astrometry.py``.

Three groups of functions live here.

Reading epoch astrometry
    ``load_epoch_astrometry`` turns the raw DR4 pre-release VOTable into a
    flat, one-row-per-CCD-observation DataFrame for a single source. It is
    backed by two readers -- ``_rows_via_astropy`` and the dependency-free
    ``_parse_binary2_votable`` -- and by ``_jd_to_jyear``.

Initializing an orbit
    ``scan_orbit_init`` runs a Thiele-Innes grid search over trial periods,
    eccentricities and periastron times to produce starting values for the
    Campbell elements, so the sampler does not have to find the orbit from
    an uninformative prior. ``_kepler_E`` solves Kepler's equation for it.

Helpers used downstream
    ``FilterRVData`` reads a radial-velocity csv and applies its optional
    ``Use`` column. ``BinByEpoch`` and ``GetModel`` are called by
    ``xo_ResultPlots.py`` when drawing the fit results.

Conventions
-----------
Angles are in radians unless the name ends in ``_deg``. Along-scan
quantities (``w``, ``w_err``, ``a0``) are in mas. Time appears on three
axes, and it matters which one a given array is on:

    ``t_jyear``  absolute Julian year, TCB (e.g. 2017.83)
    ``dt``       Julian years measured from J2017.5, the DR4 design-matrix
                 reference epoch -- the axis the astrometric model works on
    BJD          used only for radial velocities, converted by the caller

The scan position angle is stored as ``Theta`` in radians, converted from
the ``scan_pos_angle`` column in degrees. The along-scan unit vector is
``(sin Theta, cos Theta)`` in (alpha*, delta).

Gaia epoch times are on the TCB scale, which differs from TDB by ~24 s with
a slowly growing offset. No TCB-to-TDB conversion is applied anywhere in
this module.

Columns returned by load_epoch_astrometry
-----------------------------------------
    ``t_jyear``           observation time, Julian year (TCB)
    ``dt``                ``t_jyear`` - 2017.5, in Julian years
    ``Theta``             scan position angle, radians
    ``ParallaxFactorAl``  d(w)/d(parallax), dimensionless
    ``ParallaxFactorAc``  the same for the across-scan direction; all NaN
                          when the input table has no ``parallax_factor_ac``
    ``RA0``, ``Dec0``     catalogue reference position of the source, degrees
    ``w``                 along-scan centroid abscissa, mas
    ``w_err``             its formal uncertainty, mas
    ``used``              the ``used_by_agis_al`` flag
    ``transit_id``        field-of-view transit the observation belongs to;
                          used as the epoch label when binning
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import base64
import re
import struct

# ----------------------------------------------------------------------
# DR4_REF_EPOCH_JYEAR : epoch the DR4 design matrix is referred to; the
#                       'dt' column is measured from here.
# JD_J2010            : default TIMESYS time origin of the pre-release
#                       VOTable, used when the file declares none.
# DAYS_PER_JYEAR      : days in a Julian year, exact by definition.
# JD_J2000            : J2000.0 in JD, the zero point for _jd_to_jyear.
# _SZ                 : VOTable datatype -> (width in bytes, struct
#                       format) for the BINARY2 reader. 'boolean' has no
#                       struct format because it is read a byte at a time.
# ----------------------------------------------------------------------
DR4_REF_EPOCH_JYEAR = 2017.5
JD_J2010 = 2455197.5          # default TIMESYS timeorigin of the file (TCB)
DAYS_PER_JYEAR = 365.25
JD_J2000 = 2451545.0

_SZ = {"long": (8, ">q"), "double": (8, ">d"), "float": (4, ">f"),
       "short": (2, ">h"), "int": (4, ">i"), "boolean": (1, None),
       "unsignedByte": (1, ">B")}

def FilterRVData(FilePath, delimiter=','):
	"""
	Read a radial-velocity csv and drop the rows flagged as unused.

	Parameters
	----------
	FilePath : str
		Path to a csv holding every instrument. xo_Astrometry.py expects
		the columns bjd, rv [m/s], e_rv [m/s] and inst (instrument label,
		lower case), one row per measurement.
	delimiter : str, optional
		Column separator passed through to pandas, default ','.

	Returns
	-------
	pandas.DataFrame
		The input table keeping only rows whose 'Use' entry is truthy.
		If there is no 'Use' column every row is kept, so the column is
		optional. The index is not reset, so it still refers to the
		original file rows.
	"""
	df = pd.read_csv(FilePath, delimiter=delimiter)
	if np.any(df.columns == 'Use'):
		Flag = np.array(df['Use']).astype(bool)
	else:
		Flag = np.ones(len(df), dtype=bool)

	return df[Flag]

def _jd_to_jyear(jd):
    """Julian date -> Julian year. Scale-agnostic: TCB in, TCB out."""
    return 2000.0 + (np.asarray(jd, dtype=float) - JD_J2000) / DAYS_PER_JYEAR

def _rows_via_astropy(path):
    """
    Read the VOTable with astropy -> (None, list-of-dict rows, timeorigin).

    Each row is a dict of column name -> value, with masked columns
    filled with NaN so the caller never has to handle masks. The first
    element is None because astropy does not hand back the FIELD
    definitions that _parse_binary2_votable returns; both callers ignore
    it.

    The TIMESYS timeorigin is not exposed by every astropy version, so
    JD_J2010 is returned unconditionally rather than read from the file.
    A file declaring a different origin therefore needs the BINARY2
    reader, reachable with use_astropy=False.
    """
    from astropy.io.votable import parse_single_table
    table = parse_single_table(path).to_table()
    rows = []
    for r in table:
        rows.append({c: (np.ma.filled(np.ma.asarray(r[c]), np.nan)
                         if getattr(r[c], "shape", ()) else r[c])
                     for c in table.colnames})
    # TIMESYS timeorigin is not exposed by all astropy versions; use default.
    return None, rows, JD_J2010

def _parse_binary2_votable(path):
    """
    Minimal BINARY2 VOTable reader -> (fields, list-of-dict rows).

    A dependency-free fallback for _rows_via_astropy. The FIELD
    definitions are scraped from the XML header with regexes, the base64
    <STREAM> payload is decoded, and the rows are unpacked according to
    the BINARY2 layout: a per-row null mask of ceil(nfields/8) bytes,
    then each field in declaration order, with variable-length fields
    (arraysize="*") prefixed by a 4-byte count.

    The mask bytes are skipped rather than decoded, so a field flagged
    null is read as its raw placeholder value instead of as missing.
    That is harmless for this file, where the columns the loader needs
    are always populated, but it is worth knowing before pointing this
    reader at a sparser table.

    Returns
    -------
    fields : list of (name, datatype, arraysize)
    rows : list of dict
        One entry per field-of-view transit, keyed by column name.
    timeorigin : float
        The TIMESYS timeorigin declared in the file, or JD_J2010 when
        the file declares none.
    """
    txt = open(path).read()
    fields = []
    for m in re.finditer(r"<FIELD([^>]*)>", txt):
        a = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
        fields.append((a["name"], a["datatype"], a.get("arraysize")))
    m = re.search(r"<STREAM[^>]*>(.*?)</STREAM>", txt, re.S)
    if m is None:
        raise ValueError("No BINARY2 <STREAM> found in %s" % path)
    buf = base64.b64decode(m.group(1))

    tm = re.search(r'<TIMESYS[^>]*timeorigin="([^"]+)"', txt)
    timeorigin = float(tm.group(1)) if tm else JD_J2010

    nmask = (len(fields) + 7) // 8
    pos, rows = 0, []
    while pos < len(buf):
        pos += nmask  # skip the per-row null mask
        row = {}
        for name, dt, arr in fields:
            n, fmt = _SZ[dt]
            if arr == "*":
                cnt = struct.unpack(">i", buf[pos:pos + 4])[0]
                pos += 4
                if dt == "boolean":
                    vals = np.frombuffer(buf[pos:pos + cnt],
                                         dtype="S1") == b"T"
                    pos += cnt
                else:
                    # The `if False` below is dead: the dtype is always
                    # np.dtype(fmt). Left as written.
                    vals = np.frombuffer(buf[pos:pos + cnt * n],
                                         dtype=fmt.replace(">", ">") if False
                                         else np.dtype(fmt)).copy()
                    pos += cnt * n
                row[name] = np.asarray(vals)
            else:
                if dt == "boolean":
                    row[name] = buf[pos:pos + 1] == b"T"
                    pos += 1
                else:
                    row[name] = struct.unpack(fmt, buf[pos:pos + n])[0]
                    pos += n
        rows.append(row)
    return fields, rows, timeorigin


def load_epoch_astrometry(xml_path, source_id, use_astropy=True,
                          only_used_by_agis=True):
    """
    Return a flat per-CCD-observation DataFrame for one source.
      
    Loader for the Gaia DR4 pre-release epoch astrometry RAW VOTable
    (GAIA_DR4_PRERELEASE_EPOCH_ASTROMETRY_RAW.xml).

    The file is ONE VOTable (BINARY2 serialisation) containing all 12 targets.
    Each row is one field-of-view transit of one source; several columns are
    variable-length arrays with one entry per CCD observation within the transit
    (obs_time_tcb [ns since the TIMESYS time origin, JD 2455197.5 TCB],
    scan_pos_angle [deg], centroid_pos_al [mas], centroid_pos_error_al [mas],
    used_by_agis_al [bool], ...), while others are per-transit scalars
    (source_id, transit_id, parallax_factor_al, ra0, dec0, ...).

    This module parses the table, selects one source_id, explodes the per-CCD
    arrays into a flat table, applies the used_by_agis_al filter, and converts
    times to Julian years relative to the DR4 reference epoch J2017.5 --
    following the conventions of the ESA notebooks
    (esa/gaia-jupyter-notebooks: Gaia-DR4-prerelease_analyse_epoch_astrometry;
    esa/gaia-bhthree: Gaia_BH3_fit_astrometric_orbit).

    Parameters
    ----------
    xml_path : str
        Path to the epoch-astrometry VOTable.
    source_id : int
        Gaia source_id to extract. If it is not present, SystemExit is
        raised listing the source_ids that are.
    use_astropy : bool, optional
        Try the astropy reader first (default). False forces the
        pure-python BINARY2 reader. Note the fallback is unconditional:
        any exception out of the astropy path, including a malformed
        file, drops through to _parse_binary2_votable rather than
        propagating, so a silent fallback is possible.
    only_used_by_agis : bool, optional
        Keep only observations flagged used_by_agis_al, i.e. the ones
        that entered the astrometric solution. Default True.

    Returns
    -------
    pandas.DataFrame
        One row per CCD observation, with the columns listed in the
        module docstring, re-indexed 0..N-1. Rows with a non-finite w,
        w_err, Theta or ParallaxFactorAl, or with w_err <= 0, are
        dropped whatever only_used_by_agis is set to. Rows come back in
        the order the transits appear in the file, so sort on 'dt' if
        time order matters.

    A one-line summary of how many observations survived, and the time
    span they cover, is printed as a side effect.
        
    """

    # use_astropy=False raises ImportError on purpose, so that forcing the
    # BINARY2 reader reuses the same except-branch as a real failure of
    # the astropy path.
    try:
        if not use_astropy:
            raise ImportError
        _, rows, timeorigin = _rows_via_astropy(xml_path)
    except Exception:
        _, rows, timeorigin = _parse_binary2_votable(xml_path)

    ids = sorted({int(r["source_id"]) for r in rows})
    if int(source_id) not in ids:
        raise SystemExit(
            "source_id %s not in %s.\nAvailable source_ids:\n  %s"
            % (source_id, xml_path, "\n  ".join(str(i) for i in ids)))

    rec = []
    for r in rows:
        if int(r["source_id"]) != int(source_id):
            continue
        t_ns = np.asarray(r["obs_time_tcb"], dtype=float)
        nobs = len(t_ns)

        # Some columns hold one value per CCD observation and some are
        # per-transit scalars; broadcast the scalars so every column in
        # the block has nobs entries.
        def per_ccd(name):
            v = np.asarray(r[name], dtype=float)
            return v if v.ndim and len(v) == nobs else np.full(nobs, float(v))

        jd = timeorigin + t_ns * 1e-9 / 86400.0     # TCB Julian date
        jyear = _jd_to_jyear(jd)
        Theta = np.deg2rad(per_ccd("scan_pos_angle"))
        ParallaxFactorAl = per_ccd("parallax_factor_al")
        try:
            ParallaxFactorAc = per_ccd("parallax_factor_ac")
        except Exception:
            ParallaxFactorAc = np.full(nobs, np.nan)
            
        block = dict(
            t_jyear=jyear, # JD in years
            dt=jyear - DR4_REF_EPOCH_JYEAR, # JD offset from Gaia DR4 epoch = 2017.5
            Theta=Theta, # Same as psi = scan_pos_angle in radians
            ParallaxFactorAl=ParallaxFactorAl, # Derivative of along scan coordinate 'w' wrt parallax
            ParallaxFactorAc=ParallaxFactorAc, # Derivative of across scan coordinate 'z' wrt parallax

            # AL unit vector = (sin theta, cos theta) in (alpha*, delta), AC
            # unit vector = (cos psi, -sin psi)
            # f_ra=ParallaxFactorAl*np.sin(Theta) + ParallaxFactorAC*np.cos(Theta),
            # f_dec=ParallaxFactorAl*np.cos(Theta) - ParallaxFactorAC*np.sin(Theta),
            RA0=np.full(nobs, float(r["ra0"])),
            Dec0=np.full(nobs, float(r["dec0"])),
            w=per_ccd("centroid_pos_al"),
            w_err=per_ccd("centroid_pos_error_al"),
            used=np.asarray(r["used_by_agis_al"], dtype=bool),
            transit_id=np.full(nobs, int(r["transit_id"]), dtype=np.int64),
        )
        rec.append(pd.DataFrame(block))

    # Quality mask. The finite-value and positive-error cuts always
    # apply; the used_by_agis_al cut is the optional one.
    data = pd.concat(rec, ignore_index=True)
    n0 = len(data)
    good = np.isfinite(data[["w", "w_err", "Theta",
                             "ParallaxFactorAl"]].values).all(axis=1)
    good &= data["w_err"].values > 0
    if only_used_by_agis:
        good &= data["used"].values
    data = data[good].reset_index(drop=True)
    print("source %d: %d CCD observations kept (of %d), "
          "%.2f - %.2f (Julian year)"
          % (int(source_id), len(data), n0,
             data["t_jyear"].min(), data["t_jyear"].max()))
    return data


def _kepler_E(M: np.ndarray, e: float, max_iter: int = 100, tol: float = 1e-8) -> np.ndarray:
    """
    Solves Kepler's Equation: M = E - e * sin(E) using Newton-Raphson iteration.
    
    Parameters
    ----------
    M : np.ndarray
        Mean anomaly array in radians.
    e : float
        Orbital eccentricity (0 <= e < 1).
    max_iter : int, optional
        Maximum iterations for convergence, default is 100.
    tol : float, optional
        Convergence tolerance, default is 1e-8.
        
    Returns
    -------
    np.ndarray
        Eccentric anomaly (E) array in radians.

    Notes
    -----
    No error is raised if the iteration has not converged within
    max_iter; the last estimate is simply returned. In practice this
    converges well inside that budget over the range used here:
    machine precision in 7 iterations at e = 0.90 and 8 at e = 0.95,
    which is the highest eccentricity scan_orbit_init's grids reach.
    Newton-Raphson on Kepler's equation does become unreliable for e
    approaching 1 with M near zero, so raise that ceiling with care.
    """
    E = np.array(M, dtype=float)
    for _ in range(max_iter):
        delta = (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        E -= delta
        if np.all(np.abs(delta) < tol):
            break
    return E

# =============================================================================
# MAIN INITIALIZATION FUNCTION
# =============================================================================
def scan_orbit_init(data, map_soln=None, P_min=None, P_max=None, oversample=10.0,
                    top_k=5, MakePlot=True):
    """
    Hybrid Thiele-Innes orbital grid initializer for 1D along-scan astrometry.
    
    Extracts initial guesses for Campbell elements (P, e, T0, a0, i, omega, Omega)
    by running linear Thiele-Innes grid searches over candidate periods and eccentricities.

    Parameters
    ----------
    data : pandas.DataFrame
        Epoch astrometry from load_epoch_astrometry. The columns dt,
        Theta, w, w_err and ParallaxFactorAl are used.
    map_soln : dict, optional
        A single-star MAP solution (keys dRA, dDec, PMRA, PMDec,
        Parallax). If given, that model is subtracted first and only the
        four Thiele-Innes constants are solved for at each grid point.
        If None -- the default, and what xo_Astrometry.py passes -- the
        five astrometric and four orbital terms are solved jointly, nine
        linear parameters in all.
    P_min, P_max : float, optional
        Period search limits in days. Default to 10 days and twice the
        observing baseline.
    oversample : float, optional
        Frequency oversampling of the circular periodogram relative to
        1/baseline. Default 10.
    top_k : int, optional
        How many periodogram minima are carried into the eccentric
        refinement. Default 5.
    MakePlot : bool, optional
        Also build a chi2-versus-period figure. Default True.

    Returns
    -------
    init : dict
        P_days, e, T0, a0 [mas], cosi, omega, Omega [radians], phase and
        red_chi2. These are the keys xo_Astrometry.py reads when seeding
        the binary model.
    fig : matplotlib.figure.Figure
        Only meaningful when MakePlot is True. When it is False the
        second return value is whatever the name `_` happens to hold at
        that point -- the leftover counter from the refinement loop, not
        a figure. Callers that pass MakePlot=False should discard it.

    Notes
    -----
    The search runs in three stages: a linear circular periodogram over a
    frequency-uniform grid; refinement of the top_k minima over a coarse
    eccentricity grid; then up to eight zoom iterations around the best
    candidate. At fixed (P, e, T0) the Thiele-Innes constants enter
    linearly, so each grid point costs one small linear solve rather than
    a nonlinear fit.

    If the best orbit improves on the 5-parameter model by less than
    delta-chi2 = 50 for its seven extra parameters, a note is printed:
    that usually means there is no orbital signal in the astrometry and
    the 'Single' analysis is the appropriate one.
    """
    # Exact column mapping matching Index(['index', 't_jyear', 'dt', 'Theta', ...])
    t = data['dt'].values
    s = np.sin(data['Theta'].values)
    c = np.cos(data['Theta'].values)
    W = 1.0 / data['w_err'].values
    pf_al = data['ParallaxFactorAl'].values
    y_raw = data['w'].values

    # Design matrix for 5-parameter astrometric baseline: [dRA, dDec, Parallax, PMRA, PMDec]
    base = np.column_stack([s, c, pf_al, t * s, t * c])

    if map_soln is not None:
        # Subtract known single-star baseline to isolate pure orbital residuals
        w_single_star = (s * (map_soln['dRA'] + map_soln['PMRA'] * t) + 
                         c * (map_soln['dDec'] + map_soln['PMDec'] * t) + 
                         map_soln['Parallax'] * pf_al)
        y = y_raw - w_single_star 
        chi2_5p = np.sum((y * W) ** 2)
    else:
        # Simultaneous 9-parameter joint fit (5 baseline + 4 orbital)
        y = y_raw
        c5, *_ = np.linalg.lstsq(base * W[:, None], y * W, rcond=None)
        chi2_5p = np.sum(((y - base @ c5) * W) ** 2)

    # Baseline and typical visit spacing, both in days, used to set the
    # default period range and the frequency resolution below.
    baseline_d = (t.max() - t.min()) * 365.25
    tu = np.unique(np.round(np.sort(t) * 365.25))
    med_gap = np.median(np.diff(tu)) if len(tu) > 1 else np.nan
    
    if P_min is None: P_min = 10.0        
    if P_max is None: P_max = 2.0 * baseline_d

    # =========================================================================
    # INTERNAL HELPER 1: LINEAR THIELE-INNES SOLVER
    # =========================================================================
    # On the mutable default: `_cache` is rebound every time
    # scan_orbit_init runs, because the `def` statement itself
    # re-executes, so the cache is per-call and never shared between
    # calls. It memoizes the parts of the normal equations that do not
    # depend on (P, e, T0).
    def solve(X, Y, _cache={}):
        """
        Solves the weighted normal equations (O^T * W * O) * C = O^T * W * y
        to find the linear Thiele-Innes constants (A, B, F, G) for fixed (P, e, T0).
        
        Parameters
        ----------
        X, Y : np.ndarray
            Normalized elliptical coordinates in the orbital plane.
            X = cos(E) - e, Y = sqrt(1 - e^2) * sin(E).
            
        Returns
        -------
        chi2 : float
            Weighted sum of squared residuals for this linear solution.
        coef : np.ndarray
            Best-fit coefficients array containing Thiele-Innes constants.
        """
        if map_soln is not None:
            if not _cache: _cache["yy"] = np.sum((y * W) ** 2)
            Ow = np.column_stack([X * c, X * s, Y * c, Y * s]) * W[:, None]
            G = Ow.T @ Ow      
            b = Ow.T @ (y * W) 
            G[np.diag_indices(4)] += 1e-10 * np.trace(G) / 4.0
        else:
            if not _cache:
                Bw = base * W[:, None]
                _cache["Bw"] = Bw
                _cache["G00"] = Bw.T @ Bw
                _cache["b0"] = Bw.T @ (y * W)
                _cache["yy"] = np.sum((y * W) ** 2)
            Bw = _cache["Bw"]
            Ow = np.column_stack([X * c, X * s, Y * c, Y * s]) * W[:, None]
            G = np.empty((9, 9))
            G[:5, :5] = _cache["G00"]
            G[:5, 5:] = Bw.T @ Ow
            G[5:, :5] = G[:5, 5:].T
            G[5:, 5:] = Ow.T @ Ow
            b = np.concatenate([_cache["b0"], Ow.T @ (y * W)])
            G[np.diag_indices(9)] += 1e-10 * np.trace(G) / 9.0

        try:
            coef = np.linalg.solve(G, b)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(G, b, rcond=None)[0]
        return _cache["yy"] - b @ coef, coef

    # =========================================================================
    # INTERNAL HELPER 2: PERIASTRON GRID RESOLUTION SCALER
    # =========================================================================
    def nT0_for(e: float) -> int:  
        """
        Dynamically calculates the grid density needed for time of periastron (T0).
        Because highly eccentric orbits sweep through periastron rapidly, sampling 
        density must scale inversely with (1 - e) to prevent missing the signal spike.
        
        Parameters
        ----------
        e : float
            Orbital eccentricity.
            
        Returns
        -------
        int
            Number of T0 grid evaluation steps (clamped between 40 and 240).
        """
        return int(np.clip(30.0 / max(1.0 - e, 0.05), 40, 240))

    # =========================================================================
    # INTERNAL HELPER 3: GRID SEARCH SCANNER
    # =========================================================================
    def grid_best(Ps: np.ndarray, es: np.ndarray, best: tuple, nT0: int = None) -> tuple:
        """
        Iterates over parameter grids of Period (P), Eccentricity (e), and Periastron (T0)
        to identify the combination yielding the absolute minimum chi-squared.
        
        Parameters
        ----------
        Ps : np.ndarray
            Array of trial orbital periods in days.
        es : np.ndarray
            Array of trial eccentricities.
        best : tuple
            Current best known solution state: (best_chi2, (P, e, T0, coef)).
        nT0 : int, optional
            Fixed T0 resolution override. If None, uses dynamic `nT0_for(e)`.
            
        Returns
        -------
        tuple
            Updated best solution tuple: (min_chi2, (best_P, best_e, best_T0, best_coef)).
        """
        for P in Ps:
            Py = P / 365.25
            for e in es:
                n = nT0 if nT0 is not None else nT0_for(e)
                for T0 in np.linspace(-Py / 2, Py / 2, n):
                    M = np.mod(2 * np.pi * (t - T0) / Py + np.pi, 2 * np.pi) - np.pi
                    E = _kepler_E(M, e) 
                    X, Y = np.cos(E) - e, np.sqrt(1 - e * e) * np.sin(E)
                    c2, coef = solve(X, Y)
                    if c2 < best[0]:
                        best = (c2, (P, e, T0, coef))
        return best

    # -------------------------------------------------------------------------
    # STEP 1: Frequency-uniform circular periodogram
    # -------------------------------------------------------------------------
    df = 1.0 / (oversample * baseline_d)                  
    freq = np.arange(1.0 / P_max, 1.0 / P_min + df, df)
    Pgrid = 1.0 / freq[::-1]
    chi = np.empty_like(Pgrid)
    
    for i, P in enumerate(Pgrid):
        M = 2.0 * np.pi * t / (P / 365.25)
        chi[i], _ = solve(np.cos(M), np.sin(M))
            
    print("Period scan: %d frequencies, P in [%.0f, %.0f] d "
          "(baseline %.0f d, median visit gap %.1f d); "
          "best delta-chi2 over 5p model = %.0f"
          % (len(Pgrid), P_min, P_max, baseline_d, med_gap,
             chi2_5p - chi.min()))
    
    # Isolate top-K local minima separated by > 3 peak widths
    fgrid = 1.0 / Pgrid
    order = np.argsort(chi)
    peaks = []
    min_sep = 3.0 / baseline_d                    
    for i in order:
        if all(abs(fgrid[i] - fgrid[j]) > min_sep for j in peaks):
            peaks.append(i)
        if len(peaks) >= top_k:
            break

    # -------------------------------------------------------------------------
    # STEP 2: Refine candidate peaks with eccentric grid search
    # -------------------------------------------------------------------------
    e_coarse = np.linspace(0.0, 0.95, 12)
    cands = []
    for j in peaks:
        Pj = Pgrid[j]
        b = grid_best(np.linspace(0.95 * Pj, 1.05 * Pj, 9), e_coarse, (np.inf, None))
        cands.append(b)
    cands.append(grid_best(
        np.exp(np.linspace(np.log(P_min), np.log(P_max), 40)),
        np.array([0.0, 0.3, 0.6, 0.8, 0.9]), (np.inf, None), nT0=25))

    # 12 = 9 linear coefficients + the three grid parameters (P, e, T0).
    dof = float(len(data) - 12)
    cands.sort(key=lambda b: b[0])
    print("Refined period candidates (red.chi2 @ P, e):")
    for c2c, (Pc, ec, _, _) in cands:
        print("   %8.4f @ P = %8.2f d, e = %.2f" % (c2c / dof, Pc, ec))
        
    best = cands[0]
    for _ in range(8):  
        P0 = best[1][0]
        best = grid_best(np.linspace(0.85 * P0, 1.15 * P0, 15),
                         np.linspace(0.0, 0.95, 20), best)
        if abs(best[1][0] - P0) < 0.01 * P0:
            break
    c2, (P, e, T0, coef) = best

    dchi2_orbit = chi2_5p - c2
    if dchi2_orbit < 50.0:
        print("NOTE: the best orbit improves on the single-star model by "
              "only delta-chi2 = %.1f (7 extra parameters); there is no "
              "significant orbital signal in this astrometry and the "
              "'single' analysis is probably the appropriate one."
              % dchi2_orbit)

    
    if MakePlot is True:
        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.semilogx(Pgrid, chi, lw=0.7)
        ax.axvline(P, color="C1", ls="--", label=f"adopted P = {P:.1f} d")
        ax.set_xlabel("Period [days]"); ax.set_ylabel(r"$\chi^2$")
        ax.legend(); fig.tight_layout()
        # fig.savefig(plot_to, dpi=150); plt.close(fig)

    # -------------------------------------------------------------------------
    # STEP 3: Convert Thiele-Innes constants to Campbell geometric elements
    # -------------------------------------------------------------------------
    # Where the Thiele-Innes constants sit in `coef` depends on which
    # branch of solve() ran: the four constants alone when the 5p model
    # was pre-subtracted, or five baseline terms followed by the four
    # constants in the joint case.
    if map_soln is not None:
        A_, B_, F_, G_ = coef[0], coef[1], coef[2], coef[3]
    else:
        A_, B_, F_, G_ = coef[5], coef[6], coef[7], coef[8]

    # Standard Thiele-Innes -> Campbell inversion. a0 comes out in mas
    # because the constants were fit against w in mas; the two arctan2
    # combinations give omega +/- Omega, which separate into the two
    # angles.
    u = 0.5 * (A_**2 + B_**2 + F_**2 + G_**2)
    v = A_ * G_ - B_ * F_
    a0 = np.sqrt(u + np.sqrt(max(u * u - v * v, 0.0)))
    cosi = np.clip(v / a0 ** 2, -1.0, 1.0)

    # arctan2(x1, x2) = tan inverse of x1/x2
    wpO = np.arctan2(B_ - F_, A_ + G_)
    wmO = np.arctan2(-B_ - F_, A_ - G_)
    omega, Omega = (wpO + wmO) / 2.0, (wpO - wmO) / 2.0

    # Normalize angles to [0, 2*pi]
    omega = np.mod(omega, 2 * np.pi)
    Omega = np.mod(Omega, 2 * np.pi)
    
    Py = P / 365.25
    init = dict(P_days=P, e=min(e, 0.93), T0=T0, a0=a0, cosi=cosi,
                omega=omega, Omega=Omega,
                phase=np.mod(2 * np.pi * T0 / Py + np.pi, 2 * np.pi) - np.pi,
                red_chi2=c2 / dof)
    print("Orbit init: P=%.1f d e=%.2f a0=%.3f mas i=%.1f deg "
          "omega=%.1f deg Omega=%.1f deg (red.chi2=%.2f)"
          % (P, e, a0, np.degrees(np.arccos(cosi)),
             np.degrees(omega), np.degrees(Omega), init["red_chi2"]))    
    if MakePlot:
        return init, fig
    else:
        return init, _


def BinByEpoch(t, y, yerr, EpochID):
    """
    Inverse-variance weighted average of y within each unique EpochID.

    Gaia records several CCD observations per field-of-view transit, so
    the natural epoch label is transit_id. The scan angle is fixed within
    a transit, which is why averaging the along-scan value and then
    projecting onto the sky gives the same answer as projecting first and
    then averaging.

    Parameters
    ----------
    t, y, yerr : array_like
        Times, values and 1-sigma uncertainties, one entry per
        observation.
    EpochID : array_like
        Epoch label per observation; rows sharing a label are averaged.

    Returns
    -------
    tBin, yBin, eBin : ndarray
        One entry per unique epoch, sorted by time. tBin is the
        unweighted mean time of the epoch, yBin the inverse-variance
        weighted mean value, and eBin is 1/sqrt(sum(1/yerr**2)).
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)
    EpochID = np.asarray(EpochID)
 
    Unique = pd.unique(EpochID)
    tBin = np.zeros(len(Unique))
    yBin = np.zeros(len(Unique))
    eBin = np.zeros(len(Unique))
    for i, uid in enumerate(Unique):
        m = EpochID == uid
        w = 1.0 / yerr[m] ** 2
        tBin[i] = np.mean(t[m])
        yBin[i] = np.sum(w * y[m]) / np.sum(w)
        eBin[i] = 1.0 / np.sqrt(np.sum(w))
    Order = np.argsort(tBin)
    return tBin[Order], yBin[Order], eBin[Order]
 
 
def GetModel(Key, soln=None, trace=None):
    """
    Median of trace[Key], or soln[Key]. Returns (median, lo, hi); lo/hi None for MAP.

    Lets the plotting routines accept either a MAP solution or an MCMC
    trace without branching on which one they were handed.

    Parameters
    ----------
    Key : str
        Name of a free or deterministic variable in the model.
    soln : dict, optional
        MAP solution. If given it wins, and the value is returned as-is
        with no interval.
    trace : pymc3.MultiTrace, optional
        Posterior samples. The 16th, 50th and 84th percentiles are taken
        over the sample axis.

    Returns
    -------
    (median, lo, hi)
        lo and hi are None when soln was used.
    """
    if soln is not None:
        return np.asarray(soln[Key]), None, None
    Lo, Med, Hi = np.percentile(trace[Key], axis=0, q=[16, 50, 84])
    return Med, Lo, Hi
 
        