# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import base64
import re
import struct

DR4_REF_EPOCH_JYEAR = 2017.5
JD_J2010 = 2455197.5          # default TIMESYS timeorigin of the file (TCB)
DAYS_PER_JYEAR = 365.25
JD_J2000 = 2451545.0

_SZ = {"long": (8, ">q"), "double": (8, ">d"), "float": (4, ">f"),
       "short": (2, ">h"), "int": (4, ">i"), "boolean": (1, None),
       "unsignedByte": (1, ">B")}


def FilterRVData(FilePath, delimiter=','):
	df = pd.read_csv(FilePath, delimiter=delimiter)
	if np.any(df.columns == 'Use'):
		Flag = np.array(df['Use']).astype(bool)
	else:
		Flag = np.ones(len(df), dtype=bool)

	return df[Flag]

"""
Loader for the Gaia DR4 pre-release epoch astrometry RAW VOTable
(GAIA_DR4_PRERELEASE_EPOCH_ASTROMETRY_RAW.xml).

The file is ONE VOTable (BINARY2 serialisation) containing all 12 targets.
Each row is one field-of-view transit of one source; several columns are
variable-length arrays with one entry per CCD observation within the transit
(obs_time_tcb [ns since the TIMESYS time origin, JD 2455197.5 TCB],
scan_pos_angle [deg], centroid_pos_al [mas], centroid_pos_error_al [mas],
used_by_agis_al [bool], ...), while others are per-transit scalars
(source_id, transit_id, parallax_factor_al, ra0, dec0, ...).

This module parses the table (astropy if available, otherwise a minimal
pure-python BINARY2 reader), selects one source_id, explodes the per-CCD
arrays into a flat table, applies the used_by_agis_al filter, and converts
times to Julian years relative to the DR4 reference epoch J2017.5 --
following the conventions of the ESA notebooks
(esa/gaia-jupyter-notebooks: Gaia-DR4-prerelease_analyse_epoch_astrometry;
 esa/gaia-bhthree: Gaia_BH3_fit_astrometric_orbit).
"""

def _jd_to_jyear(jd):
    return 2000.0 + (np.asarray(jd, dtype=float) - JD_J2000) / DAYS_PER_JYEAR

def _rows_via_astropy(path):
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
    """Minimal BINARY2 VOTable reader -> (fields, list-of-dict rows)."""
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
    """Return a flat per-CCD-observation DataFrame for one source."""

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

    baseline_d = (t.max() - t.min()) * 365.25
    tu = np.unique(np.round(np.sort(t) * 365.25))
    med_gap = np.median(np.diff(tu)) if len(tu) > 1 else np.nan
    
    if P_min is None: P_min = 10.0        
    if P_max is None: P_max = 2.0 * baseline_d

    # =========================================================================
    # INTERNAL HELPER 1: LINEAR THIELE-INNES SOLVER
    # =========================================================================
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
    if map_soln is not None:
        A_, B_, F_, G_ = coef[0], coef[1], coef[2], coef[3]
    else:
        A_, B_, F_, G_ = coef[5], coef[6], coef[7], coef[8]

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
    """Inverse-variance weighted average of y within each unique EpochID."""
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
    """Median of trace[Key], or soln[Key]. Returns (median, lo, hi); lo/hi None for MAP."""
    if soln is not None:
        return np.asarray(soln[Key]), None, None
    Lo, Med, Hi = np.percentile(trace[Key], axis=0, q=[16, 50, 84])
    return Med, Lo, Hi
 
        