#!/usr/bin/env python3
"""
rot_alpha_first_principles.py

A self-contained, reproducible *demonstration script* for deriving the
fine-structure constant alpha from a ROT-style "first-principles" pipeline:

1) Define a dimensionless entropy-boundary action S[y] on a compact domain.
2) Solve the variational problem to get the *unique* minimizer y_e.
3) Convert y_e -> e (unit-consistent mapping Q -> e).
4) Compute alpha = e^2 / (4*pi) in natural units (ħ = c = ε0 = 1).
5) Show scheme independence: run multiple renormalization "schemes" and verify
   alpha is invariant within tolerance when expressed in the renormalized variable.

IMPORTANT:
- This is written to be GitHub-friendly and runnable without external deps
  (numpy optional; the script falls back to pure-Python).
- The "physics" is encoded as an explicit entropy-boundary functional with
  a well-posed minimization problem. If you want your *exact* ROT functional
  (from your latest derivation), replace `EntropyBoundaryModel.action()` with it.
- The scheme-independence structure (different coarse-graining maps that should
  yield the same y_e in the *renormalized* variable) is implemented and testable.

Run:
  python rot_alpha_first_principles.py --help
  python rot_alpha_first_principles.py --run_demo

Typical output:
  - y_e (minimizer)
  - e (dimensionless in natural units)
  - alpha
  - scheme invariance report

License: MIT (you can paste this into your repo)
"""

from __future__ import annotations
import math
import argparse
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

try:
    import numpy as np
    HAVE_NUMPY = True
except Exception:
    HAVE_NUMPY = False


# -----------------------------
# Utilities (minimal + stable)
# -----------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def linspace(a: float, b: float, n: int) -> List[float]:
    if n < 2:
        return [a]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]

def dot(u: List[float], v: List[float]) -> float:
    return sum(ui * vi for ui, vi in zip(u, v))

def l2norm(u: List[float]) -> float:
    return math.sqrt(dot(u, u))

def finite_diff_grad(f: Callable[[List[float]], float], x: List[float], eps: float) -> List[float]:
    # Central finite differences
    g = [0.0] * len(x)
    for i in range(len(x)):
        x1 = x[:]
        x2 = x[:]
        x1[i] += eps
        x2[i] -= eps
        g[i] = (f(x1) - f(x2)) / (2.0 * eps)
    return g

def project_box(x: List[float], lo: float, hi: float) -> List[float]:
    return [clamp(xi, lo, hi) for xi in x]


# -----------------------------
# ROT "first principles" core
# -----------------------------

@dataclass
class ROTConstants:
    """
    Keep everything dimensionless inside the action.
    If you already have your 5-postulate constant set, plug them here.
    """
    # "delta_anom" is known exactly 1 in your current ROT work.
    delta_anom: float = 1.0

    # Domain (dimensionless) for the boundary problem
    u_min: float = 0.0
    u_max: float = 1.0

    # Regularization (dimensionless)
    eps: float = 1e-12

@dataclass
class Scheme:
    """
    A renormalization/coarse-graining "scheme" is a map that:
      - builds a discretized representation of y(u)
      - defines how gradients/penalties are evaluated
    The goal: y_e in the properly renormalized variable should be invariant.
    """
    name: str
    # window / smoothing strength for scheme
    smooth: float
    # derivative stencil scale
    dscale: float


class EntropyBoundaryModel:
    """
    This is the place to encode your ROT entropy-boundary functional.

    We implement a strict, well-posed variational action:
      S[y] = ∫ [ (1/2) A(u) (y')^2 + V(y,u) ] du  +  boundary_terms

    with:
      - A(u) > 0   (elliptic / convex kinetic term)
      - V convex near the minimizer to ensure uniqueness

    The minimizer y_e is then mapped to a charge scale e by a unit-consistent map.
    """

    def __init__(self, C: ROTConstants):
        self.C = C

    def A(self, u: float) -> float:
        # Positive "stiffness" profile; keep smooth and bounded away from 0.
        # Replace with ROT-derived stiffness if you have it.
        return 1.0 + 0.25 * math.cos(2.0 * math.pi * u)

    def V(self, y: float, u: float) -> float:
        # Convex potential around a preferred y* determined by entropy balance.
        # Replace with your ROT-derived V(κ,η) reduced to y.
        #
        # Here: y*(u) is set by a mild profile + exact delta_anom shift.
        y_star = 0.30 + 0.02 * math.sin(2.0 * math.pi * u) + 0.01 * self.C.delta_anom
        lam = 4.0  # curvature strength (dimensionless)
        return 0.5 * lam * (y - y_star) ** 2

    def boundary_term(self, y0: float, y1: float) -> float:
        # Boundary regularizer (Neumann-ish penalty) to avoid edge artifacts.
        # Replace with your boundary entropy cell term if desired.
        return 0.25 * (y0 - y1) ** 2

    def action_discrete(self, y: List[float], u: List[float], scheme: Scheme) -> float:
        """
        Discretized action with scheme-dependent smoothing & derivative scaling.
        y corresponds to y(u) sampled on grid u.
        """
        C = self.C
        n = len(u)
        if n != len(y):
            raise ValueError("y and u lengths must match")

        # Optional scheme smoothing (simple 3-point blur applied once)
        y_eff = y[:]
        if scheme.smooth > 0.0:
            s = scheme.smooth
            y_eff2 = y_eff[:]
            for i in range(1, n - 1):
                y_eff2[i] = (1 - s) * y_eff[i] + 0.5 * s * (y_eff[i - 1] + y_eff[i + 1])
            y_eff = y_eff2

        # Kinetic + potential integral via trapezoid rule
        S = 0.0
        for i in range(n - 1):
            du = (u[i + 1] - u[i]) * scheme.dscale
            if du <= 0:
                continue

            # derivative y' ~ (y_{i+1}-y_i)/du_phys
            dy = (y_eff[i + 1] - y_eff[i])
            yprime = dy / (du + C.eps)

            ui_mid = 0.5 * (u[i] + u[i + 1])
            yi_mid = 0.5 * (y_eff[i] + y_eff[i + 1])

            A = self.A(ui_mid)
            V = self.V(yi_mid, ui_mid)

            S += (0.5 * A * yprime * yprime + V) * du

        # Boundary term
        S += self.boundary_term(y_eff[0], y_eff[-1])
        return S

    # -----------------------------
    # Unit-consistent mapping Q -> e
    # -----------------------------
    def y_to_e(self, y_e: float) -> float:
        """
        Clean mapping from the dimensionless minimizer y_e to charge e in natural units.

        In natural units (ħ = c = ε0 = 1):
          alpha = e^2 / (4*pi)

        We encode Q -> e as a monotone map:
          e = exp(-y_e)   (example)
        Replace this with your derived mapping once frozen.
        """
        return math.exp(-y_e)

    def e_to_alpha(self, e: float) -> float:
        return (e * e) / (4.0 * math.pi)


# -----------------------------
# Optimizer: projected gradient
# -----------------------------

def minimize_projected(
    f: Callable[[List[float]], float],
    x0: List[float],
    lo: float,
    hi: float,
    steps: int = 600,
    lr: float = 0.05,
    fd_eps: float = 1e-5,
    tol: float = 1e-10,
    seed: int = 0,
) -> Tuple[List[float], float, Dict[str, float]]:
    random.seed(seed)

    x = project_box(x0[:], lo, hi)
    fx = f(x)
    best = (x[:], fx)

    # Simple backtracking on divergence
    lr_now = lr
    prev_fx = fx

    for k in range(steps):
        g = finite_diff_grad(f, x, fd_eps)
        gnorm = l2norm(g) + 1e-30

        # step
        x_new = [xi - lr_now * gi for xi, gi in zip(x, g)]
        x_new = project_box(x_new, lo, hi)
        fx_new = f(x_new)

        if fx_new <= fx:
            x, fx = x_new, fx_new
            if fx < best[1]:
                best = (x[:], fx)
            # mild lr increase
            lr_now = min(lr_now * 1.01, lr)
        else:
            # backtrack
            lr_now *= 0.5

        if abs(prev_fx - fx) < tol:
            break
        prev_fx = fx

    info = {
        "final_lr": lr_now,
        "best_f": best[1],
    }
    return best[0], best[1], info


# -----------------------------
# Scheme invariance experiment
# -----------------------------

def solve_y_e_for_scheme(
    model: EntropyBoundaryModel,
    scheme: Scheme,
    grid_n: int,
    y_init: float,
) -> Tuple[float, float, Dict[str, float]]:
    C = model.C
    u = linspace(C.u_min, C.u_max, grid_n)

    # Decision variable is y(u) discretized on grid
    x0 = [y_init for _ in range(grid_n)]

    def f(x: List[float]) -> float:
        return model.action_discrete(x, u, scheme)

    x_star, f_star, info = minimize_projected(
        f=f,
        x0=x0,
        lo=-2.0,
        hi=+2.0,
        steps=900,
        lr=0.08,
        fd_eps=5e-6,
        tol=1e-12,
        seed=0,
    )

    # Define y_e as the *renormalized* invariant: average over interior
    # (avoids boundary artifacts in discrete schemes).
    interior = x_star[1:-1] if grid_n >= 3 else x_star
    y_e = sum(interior) / max(1, len(interior))

    e = model.y_to_e(y_e)
    alpha = model.e_to_alpha(e)

    # Attach results
    info2 = dict(info)
    info2.update({"y_e": y_e, "e": e, "alpha": alpha, "S": f_star})
    return y_e, alpha, info2


def scheme_set() -> List[Scheme]:
    # A small family of schemes that should agree after renormalized extraction.
    return [
        Scheme("scheme_A_minimal", smooth=0.00, dscale=1.00),
        Scheme("scheme_B_smooth",  smooth=0.10, dscale=1.00),
        Scheme("scheme_C_stiffer", smooth=0.05, dscale=1.10),
        Scheme("scheme_D_softer",  smooth=0.05, dscale=0.90),
        Scheme("scheme_E_moreSmooth", smooth=0.20, dscale=1.00),
    ]


# -----------------------------
# CLI
# -----------------------------

def run_demo(grid_n: int, y_init: float, tol: float) -> int:
    C = ROTConstants()
    model = EntropyBoundaryModel(C)

    schemes = scheme_set()
    results = []

    for sc in schemes:
        y_e, alpha, info = solve_y_e_for_scheme(model, sc, grid_n=grid_n, y_init=y_init)
        results.append((sc.name, y_e, alpha, info))

    # Report
    alphas = [a for _, _, a, _ in results]
    alpha_mean = sum(alphas) / len(alphas)
    alpha_max_dev = max(abs(a - alpha_mean) for a in alphas)

    print("\nROT alpha derivation demo (variational entropy-boundary)")
    print("--------------------------------------------------------")
    print(f"Grid points: {grid_n}")
    print(f"Mapping: e = exp(-y_e), alpha = e^2/(4*pi) [natural units]")
    print("")
    for name, y_e, alpha, info in results:
        print(f"{name:>18s} | y_e={y_e:.12f} | alpha={alpha:.12e} | S={info['S']:.12e}")

    print("")
    print(f"alpha_mean     = {alpha_mean:.12e}")
    print(f"alpha_max_dev  = {alpha_max_dev:.12e}")
    print(f"pass_invariance (<= {tol:.1e}) = {alpha_max_dev <= tol}")

    # Return code for CI
    return 0 if alpha_max_dev <= tol else 2


def main() -> int:
    ap = argparse.ArgumentParser(
        description="ROT-style derivation of alpha from a variational entropy-boundary principle + scheme-invariance test."
    )
    ap.add_argument("--run_demo", action="store_true", help="Run the full scheme-invariance demo.")
    ap.add_argument("--grid_n", type=int, default=256, help="Discretization points for y(u).")
    ap.add_argument("--y_init", type=float, default=0.25, help="Initial guess for y(u).")
    ap.add_argument("--tol", type=float, default=5e-6, help="Scheme invariance tolerance on alpha.")
    args = ap.parse_args()

    if args.run_demo:
        return run_demo(grid_n=args.grid_n, y_init=args.y_init, tol=args.tol)

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
