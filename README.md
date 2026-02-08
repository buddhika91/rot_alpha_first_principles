# ROT Derivation of the Fine-Structure Constant (α)

This repository provides a **fully reproducible, first-principles demonstration** of how the fine-structure constant  
\[
\alpha = \frac{e^2}{4\pi}
\]
can be derived from **Recursive Observation Theory (ROT)** using a **variational entropy-boundary principle**, rather than being inserted or fitted.

The code is designed to be:
- **Deterministic**
- **Unit-consistent**
- **Renormalization-scheme independent**
- **Transparent and inspectable**

It is intended as a *proof-of-principle physics demonstration*, suitable for public scrutiny and extension.

---

## Conceptual Overview

The derivation follows four logically distinct steps:

### 1. Entropy–Boundary Action (ROT First Principles)

We define a **dimensionless entropy-boundary action**
\[
S[y] = \int \left[\frac{1}{2}A(u)(y'(u))^2 + V(y,u)\right]\,du + S_{\text{boundary}}
\]

- \( y(u) \) is a dimensionless entropy-resolution field derived from ROT
- \( A(u) \) is a positive stiffness profile
- \( V(y,u) \) encodes entropy balance and collapse stability
- Boundary terms enforce well-posedness

This action is **strictly convex**, guaranteeing a unique minimizer.

---

### 2. Variational Principle → Unique Minimizer

The code solves:
\[
\frac{\delta S}{\delta y} = 0
\]

numerically via **projected gradient descent**, yielding a unique, scheme-independent minimizer:
\[
y_e = \arg\min S[y]
\]

To avoid discretization artifacts, the physical invariant is extracted as a **renormalized interior average**, not raw grid values.

---

### 3. Unit-Consistent Mapping: \( Q \rightarrow e \)

ROT predicts a dimensionless charge scale via a monotone mapping:
\[
e = \exp(-y_e)
\]

This mapping is:
- Dimensionless
- Scheme independent
- Replaceable with an exact closed-form ROT mapping once frozen

---

### 4. Prediction of the Fine-Structure Constant

In natural units \( (\hbar = c = \varepsilon_0 = 1) \):
\[
\alpha = \frac{e^2}{4\pi}
\]

The code computes α **without fitting**, **without inserting experimental values**, and **without tuning**.

---

## Scheme Independence (Renormalization Test)

A core requirement of ROT is that physical predictions must be **invariant under admissible coarse-graining schemes**.

The demo runs the full derivation across multiple schemes that vary:
- Smoothing strength
- Derivative scaling
- Numerical discretization behavior

**Only the renormalized variable is compared**, ensuring the invariance test is physically meaningful.

The script reports:
- Per-scheme \( y_e \)
- Per-scheme \( \alpha \)
- Maximum deviation across schemes

This acts as a **numerical RG fixed-point test**.

---

## Repository Structure

