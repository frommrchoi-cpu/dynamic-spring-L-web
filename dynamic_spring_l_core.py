# dynamic_spring_l_core.py
# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

def _to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna()

def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot

def _format_poly_equation(coefs: np.ndarray, var: str = "β") -> str:
    deg = len(coefs) - 1
    terms = []
    for i, c in enumerate(coefs):
        p = deg - i
        if abs(c) < 1e-12:
            continue
        c_str = f"{c:.10g}"
        if p == 0:
            terms.append(f"{c_str}")
        elif p == 1:
            terms.append(f"{c_str}·{var}")
        else:
            terms.append(f"{c_str}·{var}^{p}")
    if not terms:
        return "aL = 0"
    return ("aL = " + " + ".join(terms)).replace("+ -", "- ")

@dataclass
class RegressionResult:
    degree: int
    r2: float
    coefs: np.ndarray
    equation: str

class ALInterpolator:
    def __init__(self, calibration_csv_path: str):
        self.raw_df = self._load_calibration(calibration_csv_path)
        self.curves = self._build_curves(self.raw_df)  # S -> (B, aL, PCHIP)
        self.s_levels = sorted(self.curves.keys())
        if len(self.s_levels) < 2:
            raise ValueError(f"S/2Ro 곡선 최소 2개 필요. 현재: {self.s_levels}")

        all_b = np.concatenate([self.curves[s][0] for s in self.s_levels])
        self.beta_min = float(np.min(all_b))
        self.beta_max = float(np.max(all_b))

    @staticmethod
    def _load_calibration(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        rename = {}
        for c in df.columns:
            k = c.strip().lower()
            if k in ["s/2ro", "s/2r0", "s_2ro", "s_2r0", "s_over_2r0", "s_over_2ro"]:
                rename[c] = "S/2Ro"
            elif k in ["b", "beta", "β"]:
                rename[c] = "B"
            elif k in ["al", "a_l", "al".lower()]:
                rename[c] = "aL"
        if rename:
            df = df.rename(columns=rename)

        required = {"S/2Ro", "B", "aL"}
        if not required.issubset(df.columns):
            raise ValueError(f"보정 CSV에 {required} 열 필요. 현재: {list(df.columns)}")

        return _to_numeric_df(df[["S/2Ro", "B", "aL"]])

    @staticmethod
    def _build_curves(df: pd.DataFrame) -> Dict[float, Tuple[np.ndarray, np.ndarray, PchipInterpolator]]:
        curves = {}
        for s, g in df.groupby("S/2Ro"):
            g = g.sort_values("B")
            b = g["B"].to_numpy(float)
            al = g["aL"].to_numpy(float)
            _, idx = np.unique(b, return_index=True)
            b, al = b[idx], al[idx]
            if len(b) < 4:
                raise ValueError(f"S/2Ro={s} 점 부족(>=4). 현재 {len(b)}")
            curves[float(s)] = (b, al, PchipInterpolator(b, al, extrapolate=True))
        return curves

    def _bracket_s(self, s: float) -> Tuple[float, float]:
        L = self.s_levels
        if s <= L[0]:
            return L[0], L[1]
        if s >= L[-1]:
            return L[-2], L[-1]
        for i in range(len(L) - 1):
            if L[i] <= s <= L[i + 1]:
                return L[i], L[i + 1]
        return L[-2], L[-1]

    def aL_at(self, s_over_2r0: float, beta: float) -> float:
        s = float(s_over_2r0)
        b = float(beta)
        s1, s2 = self._bracket_s(s)
        _, _, f1 = self.curves[s1]
        _, _, f2 = self.curves[s2]
        al1 = float(f1(b))
        al2 = float(f2(b))
        if abs(s2 - s1) < 1e-12:
            return al1
        w = (s - s1) / (s2 - s1)
        return (1 - w) * al1 + w * al2

    def curve_at(self, s_over_2r0: float, betas: np.ndarray) -> np.ndarray:
        betas = np.asarray(betas, float)
        return np.array([self.aL_at(s_over_2r0, bb) for bb in betas], float)

    def curve_at_existing_s(self, s_level: float, betas: np.ndarray) -> np.ndarray:
        s_level = float(s_level)
        if s_level not in self.curves:
            raise ValueError(f"S/2r0={s_level} 곡선이 CSV에 없음. 현재: {self.s_levels}")
        _, _, f = self.curves[s_level]
        betas = np.asarray(betas, float)
        return np.array([float(f(bb)) for bb in betas], float)

    def regression_for_curve(self, s_over_2r0: float, beta_grid: np.ndarray,
                             target_r2: float = 0.99, max_degree: int = 12) -> RegressionResult:
        x = np.asarray(beta_grid, float)
        y = self.curve_at(s_over_2r0, x)
        best = None
        for deg in range(2, max_degree + 1):
            coefs = np.polyfit(x, y, deg)
            yhat = np.polyval(coefs, x)
            r2 = _r2_score(y, yhat)
            res = RegressionResult(degree=deg, r2=float(r2), coefs=coefs,
                                   equation=_format_poly_equation(coefs, var="β"))
            if best is None or res.r2 > best.r2:
                best = res
            if res.r2 >= target_r2:
                return res
        return best
