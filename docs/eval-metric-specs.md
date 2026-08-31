# Evaluation Metric Specifications & Algorithmic Pseudocode
**Author:** Pratyushi (Research Track γ — R3)  
**Recipient:** Aakash (Build Track D1 — Grid Engine & Evaluation Harness)  
**Status:** Frozen & Handed Over for Implementation (`src/eval/`)  
**Date:** 2026-08-29 (Day 1)  
**References:** `docs/sih-math.md` §8, §9, §10; `docs/master-v4.md` §2.1, §3.8

---

## 1. Executive Summary & Purpose

This document provides the exact algorithmic pseudocode, mathematical formulations, and unit test assertions for the evaluation harness in `src/eval/`:
1. **Plan Regret $R(S)$ & Discrete Fréchet Distance $d_F$** (`src/eval/plan_regret.py`)
2. **Coarsening-Justification Ratio $\rho$** (`src/eval/metrics.py`)
3. **Per-Ring Height RMSE & Occupancy IoU** (`src/eval/metrics.py`)
4. **Dynamic Removal Metrics ($DR, SP, F$)** (`src/eval/metrics.py`)
5. **Reference Map Builder & Loader** (`src/eval/reference_map.py`)

---

## 2. Plan Regret & Path Sensitivity (`src/eval/plan_regret.py`)

### 2.1 Theoretical Foundation (Math §8.1)
Let $M^*$ be the 5 cm static reference map. Let $M_S$ be the adaptive grid evaluated under schedule $S$ (e.g., 5/10/20/40 cm).
Let $P(M, \mathbf{x}_{\text{start}}, \mathbf{x}_{\text{goal}})$ be a deterministic path planner (2D 8-connected A* with diagonal Euclidean weighting).

- Optimal reference path: $\pi^* = P(M^*, \mathbf{x}_{\text{start}}, \mathbf{x}_{\text{goal}})$
- Schedule-$S$ path: $\pi_S = P(M_S, \mathbf{x}_{\text{start}}, \mathbf{x}_{\text{goal}})$

**Cost Functional $J_{M}(\pi)$:**
$$J_M(\pi) = \sum_{k=1}^{|\pi|-1} \text{Cost}_M(\mathbf{p}_k, \mathbf{p}_{k+1})$$
where for a step from $\mathbf{p}_k$ to $\mathbf{p}_{k+1}$:
$$\text{Cost}_M(\mathbf{p}_k, \mathbf{p}_{k+1}) = \|\mathbf{p}_{k+1} - \mathbf{p}_k\|_2 \cdot \left[ 1.0 + w_{\text{slope}} \cdot \text{slope}_M(\mathbf{p}_{k+1}) + w_{\text{rough}} \cdot \sigma_{z, M}^2(\mathbf{p}_{k+1}) \right]$$
If cell at $\mathbf{p}_{k+1}$ is untraversable on map $M$ (i.e. `traversability != 0`), $\text{Cost}_M = \infty$.

> ⚠️ **CRITICAL INVARIANT:** Both paths $\pi^*$ and $\pi_S$ MUST be scored on the **reference map $M^*$**:
> $$R(S) = J_{M^*}(\pi_S) - J_{M^*}(\pi^*) \ge 0$$
> Never score $\pi_S$ on $M_S$. If $\pi_S$ cuts through an obstacle/kerb that was blurred out on $M_S$, $J_{M^*}(\pi_S) = \infty \implies R(S) = \infty$ (fatal plan failure).

---

### 2.2 Algorithm 1: Plan Regret Computation

```python
def compute_plan_regret(reference_map, compressed_map, start_xy, goal_xy, w_slope=1.0, w_rough=2.0):
    """
    Args:
        reference_map: 5 cm ground-truth reference map M*
        compressed_map: Adaptive variable-resolution map M_S
        start_xy: tuple (x, y) in metres
        goal_xy: tuple (x, y) in metres
    Returns:
        dict: {
            "regret": float (R(S)),
            "cost_ref": float (J_{M*}(pi*)),
            "cost_compressed_on_ref": float (J_{M*}(pi_S)),
            "frechet_dist": float (d_F(pi_S, pi*)),
            "success": bool
        }
    """
    # Step 1: Plan optimal path on reference map
    path_star = a_star_planner(reference_map, start_xy, goal_xy)
    if path_star is None:
        raise ValueError("Start and goal disconnected on reference map!")
        
    # Step 2: Plan path on compressed/adaptive map
    path_S = a_star_planner(compressed_map, start_xy, goal_xy)
    if path_S is None:
        return {
            "regret": float("inf"),
            "cost_ref": evaluate_path_cost(reference_map, path_star, w_slope, w_rough),
            "cost_compressed_on_ref": float("inf"),
            "frechet_dist": float("inf"),
            "success": False
        }

    # Step 3: Score BOTH paths strictly on reference_map M*
    cost_star = evaluate_path_cost(reference_map, path_star, w_slope, w_rough)
    cost_S_on_ref = evaluate_path_cost(reference_map, path_S, w_slope, w_rough)
    
    # Step 4: Regret is the difference
    regret_val = max(0.0, cost_S_on_ref - cost_star)
    
    # Step 5: Geometric Fréchet distance
    d_F = discrete_frechet_distance(path_S, path_star)
    
    return {
        "regret": regret_val,
        "cost_ref": cost_star,
        "cost_compressed_on_ref": cost_S_on_ref,
        "frechet_dist": d_F,
        "success": (cost_S_on_ref < float("inf"))
    }
```

---

### 2.3 Algorithm 2: Discrete Fréchet Distance ($d_F$)

```python
def discrete_frechet_distance(path_p, path_q):
    """
    Computes exact discrete Fréchet distance d_F(P, Q) using dynamic programming.
    path_p: list of (x, y) tuples [p_1, ..., p_n]
    path_q: list of (x, y) tuples [q_1, ..., q_m]
    """
    n, m = len(path_p), len(path_q)
    ca = np.full((n, m), -1.0, dtype=np.float64)

    def dist(p, q):
        return np.hypot(p[0] - q[0], p[1] - q[1])

    def c(i, j):
        if ca[i, j] > -0.5:
            return ca[i, j]
        d = dist(path_p[i], path_q[j])
        if i == 0 and j == 0:
            ca[i, j] = d
        elif i > 0 and j == 0:
            ca[i, j] = max(c(i - 1, 0), d)
        elif i == 0 and j > 0:
            ca[i, j] = max(c(0, j - 1), d)
        else:
            ca[i, j] = max(min(c(i - 1, j), c(i, j - 1), c(i - 1, j - 1)), d)
        return ca[i, j]

    return c(n - 1, m - 1)
```

---

## 3. Information Loss & Coarsening Ratio $\rho$ (`src/eval/metrics.py`)

### 3.1 Mathematical Formulation (Math §9.3)
For each coarse cell $c \in C_L$ (cell size $c_L$), let $F(c)$ be the set of $(c_L / 0.05)^2$ constituent 5 cm reference cells subsumed by $c$.
Let $h^*_f$ be the reference height at fine cell $f \in F(c)$.

- **Reference mean:** $\bar{h}^*(c) = \frac{1}{|F(c)|} \sum_{f \in F(c)} h^*_f$
- **Sub-cell terrain spread (intrinsic variance):**
  $$\text{spread}(c) = \sqrt{\frac{1}{|F(c)|} \sum_{f \in F(c)} (h^*_f - \bar{h}^*(c))^2}$$
- **Total Information Loss:**
  $$IL(c) = \sqrt{\frac{1}{|F(c)|} \sum_{f \in F(c)} (\mu_c - h^*_f)^2} = \sqrt{(\mu_c - \bar{h}^*(c))^2 + \text{spread}(c)^2}$$
- **Coarsening-Justification Ratio:**
  $$\rho(c) = \frac{IL(c)}{\max(\text{spread}(c), \epsilon)}$$
  with $\epsilon = 0.005\text{ m}$ (5 mm floor to prevent division by zero on flat surfaces).

---

### 3.2 Algorithm 3: Coarsening Ratio Calculation

```python
def coarsening_ratio_per_ring(grid, reference_map, ring_id, eps=0.005):
    """
    Computes rho = IL / spread per ring.
    Returns:
        mean_rho: float (dimensionless ratio, rho ~ 1.0 is optimal)
        mean_spread_m: float (metres)
        mean_bias_m: float (metres)
        mean_il_m: float (metres)
    """
    cells = grid.get_ring_cells(ring_id)
    ratios, spreads, biases, ils = [], [], [], []

    for c in cells:
        if c.obs_count < 1:
            continue
        fine_cells = reference_map.get_subsumed_cells(c.x, c.y, c.size)
        if len(fine_cells) == 0:
            continue
            
        h_star = np.array([f.ground_height_m for f in fine_cells])
        h_bar_star = np.mean(h_star)
        
        spread_c = np.std(h_star)
        bias_c = abs(c.ground_height_m - h_bar_star)
        il_c = np.sqrt(bias_c**2 + spread_c**2)
        
        denom = max(spread_c, eps)
        rho_c = il_c / denom if spread_c >= eps else 1.0 + (bias_c / eps)
        
        ratios.append(rho_c)
        spreads.append(spread_c)
        biases.append(bias_c)
        ils.append(il_c)

    return {
        "mean_rho": float(np.mean(ratios)) if ratios else 1.0,
        "mean_spread_m": float(np.mean(spreads)) if spreads else 0.0,
        "mean_bias_m": float(np.mean(biases)) if biases else 0.0,
        "mean_il_m": float(np.mean(ils)) if ils else 0.0
    }
```

---

## 4. Per-Ring Reconstruction & Occupancy Metrics (`src/eval/metrics.py`)

### 4.1 Per-Ring Height RMSE (Math §9.2)
$$\text{RMSE}_L = \sqrt{\frac{1}{|C_L|} \sum_{c \in C_L} (\mu_c - \bar{h}^*(c))^2}$$

### 4.2 Per-Ring Occupancy IoU
For each state $k \in \{\text{FREE}, \text{OCCUPIED}\}$:
$$\text{IoU}_L(k) = \frac{|\{c \in C_L : \text{state}(c) = k \land \text{state}^*(c) = k\}|}{|\{c \in C_L : \text{state}(c) = k \lor \text{state}^*(c) = k\}|}$$

---

## 5. Dynamic Ghost Removal Metrics ($DR, SP, F$) (`src/eval/metrics.py`)

### 5.1 Formulation (Math §9.4)
Given a dataset sequence with ground-truth semantic point labels:
- $\mathcal{P}_{\text{dynamic}}$: LiDAR returns from moving instances (`moving-car`, `moving-bicyclist`, `moving-pedestrian`, etc.).
- $\mathcal{P}_{\text{static}}$: LiDAR returns from permanent structures (`road`, `building`, `fence`, `pole`, `vegetation`).

1. **Dynamic Removal Rate ($DR$):**
   $$DR = \frac{\sum_{p \in \mathcal{P}_{\text{dynamic}}} \mathbb{I}(\text{grid.query}(p.x, p.y).\text{occupancy} \neq \text{OCC\_OCCUPIED})}{|\mathcal{P}_{\text{dynamic}}|}$$
2. **Static Preservation Rate ($SP$):**
   $$SP = \frac{\sum_{p \in \mathcal{P}_{\text{static}}} \mathbb{I}(\text{grid.query}(p.x, p.y).\text{occupancy} == \text{OCC\_OCCUPIED})}{|\mathcal{P}_{\text{static}}|}$$
3. **Harmonic Mean ($F$-Score):**
   $$F = 2 \cdot \frac{DR \cdot SP}{DR + SP + \epsilon}$$

---

## 6. Unit Test Assertions for Aakash

Ensure `tests/test_plan_regret.py` and `tests/test_metrics.py` enforce these invariants:

1. **Test Regret Non-Negativity:** `assert regret(M_star, M_S, start, goal)["regret"] >= 0.0`
2. **Test Zero Regret on Flat Ground:** On a synthetic flat map, assert `R(S) == 0.0` and `d_F == 0.0`.
3. **Test Synthetic Kerb Resolution:** 
   - Build a 12 cm step kerb obstacle.
   - For schedule with 5 cm Ring 0: `R(S) == 0.0`.
   - For uniform 40 cm coarse map: `R(S) > 0.0` or `inf` (coarse map plans over the kerb).
4. **Test Coarsening Ratio:**
   - On flat ground with noise: assert `rho ~= 1.0` within 10%.
5. **Test Dynamic Removal:**
   - Empty grid: $DR = 1.0, SP = 0.0 \implies F = 0.0$ (proves DR alone cannot be gamed).
