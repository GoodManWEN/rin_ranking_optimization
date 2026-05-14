import numpy as np
import pandas as pd
import random
import math
import time
from numba import njit
from .utils import build_inversion_matrix, compute_objective, compute_lower_bound




# ---------------------------------------------------------------------------
# Incremental delta calculations  (hot path – numba-accelerated)
# ---------------------------------------------------------------------------

@njit
def compute_delta_swap(perm, L, p, q):
    """Delta-F when swapping perm[p] and perm[q], with p < q."""
    i = perm[p]
    j = perm[q]
    delta = L[j, i] - L[i, j]
    for r in range(p + 1, q):
        k = perm[r]
        delta += (L[k, i] - L[i, k]) + (L[j, k] - L[k, j])
    return delta

@njit
def compute_delta_insert(perm, L, p, q):
    """Delta-F when extracting perm[p] and re-inserting at position q."""
    i = perm[p]
    delta = 0.0
    if p < q:
        for r in range(p + 1, q + 1):
            k = perm[r]
            delta += L[k, i] - L[i, k]
    elif p > q:
        for r in range(q, p):
            k = perm[r]
            delta += L[i, k] - L[k, i]
    return delta
# ---------------------------------------------------------------------------
# Permutation mutation helpers  (numpy, in-place)
# ---------------------------------------------------------------------------

@njit
def _apply_insert(perm, p, q):
    """Move element at position p to position q, shifting others."""
    elem = perm[p]
    if p < q:
        for r in range(p, q):
            perm[r] = perm[r + 1]
    else:
        for r in range(p, q, -1):
            perm[r] = perm[r - 1]
    perm[q] = elem


# ---------------------------------------------------------------------------
# Initial temperature estimation  (FIXED: short-range + conservative)
# ---------------------------------------------------------------------------
 
def estimate_initial_temperature(perm, L, n_samples=500, rng=None):
    """
    Estimate T0 by sampling SHORT-range moves only.
 
    Key insight: when starting from an already-optimised permutation (after
    Borda + insertion sort), the landscape near the current point has small
    deltas.  Sampling large-distance moves (as the old version did) gives
    huge positive deltas → absurdly high T0 → the good solution is destroyed
    before any useful cooling happens.
 
    Fix: sample distances 1–10 only, use the 25th-percentile positive delta,
    and target ~30 % initial acceptance.
    """
    if rng is None:
        rng = random.Random()
    n = len(perm)
    if n <= 1:
        return 1.0
 
    max_sample_dist = min(10, n - 1)
 
    positive_deltas = []
    for _ in range(n_samples):
        p = rng.randint(0, n - 2)
        dist = rng.randint(1, min(max_sample_dist, n - 1 - p))
        q = p + dist
        delta = compute_delta_swap(perm, L, p, q)
        if delta > 0:
            positive_deltas.append(delta)
 
    if not positive_deltas:
        return 1.0
 
    positive_deltas.sort()
    idx = max(0, len(positive_deltas) // 4)          # 25th percentile
    target_delta = positive_deltas[idx]
    T0 = target_delta / (-math.log(0.3))              # 30 % acceptance
    return T0






def simulated_annealing(
    df_sub: pd.DataFrame,
    perm_init: np.ndarray | None = None,
    alpha: float = 0.97,
    markov_factor: int = 10,
    t_end_ratio: float = 0.001,
    swap_ratio: float = 0.7,
    seed: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Simulated-annealing optimisation over a single SCC's permutation.
 
    Parameters
    ----------
    df_sub : pd.DataFrame
        rank_1…rank_m columns only; index order = initial permutation.
    perm_init : np.ndarray, optional
        Override initial permutation (internal indices).
    alpha : float
        Geometric cooling factor (default 0.97).
    markov_factor : int
        Markov chain length = markov_factor × n (default 10).
    t_end_ratio : float
        Stopping temperature = T0 × t_end_ratio (default 0.001).
    swap_ratio : float
        Probability of choosing swap vs. extract-insert (default 0.7).
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool
        Print progress every ~10 % of stages.
 
    Returns
    -------
    pd.DataFrame
        Same columns as df_sub, rows reordered to the best permutation found.
    """
    rng = random.Random(seed)
 
    # --- build L and initial permutation ---
    L, unis = build_inversion_matrix(df_sub)
    n = len(unis)
    m = len(df_sub.columns)
    uni_to_idx = {u: i for i, u in enumerate(unis)}
 
    if perm_init is not None:
        perm = perm_init.copy()
    else:
        perm = np.array([uni_to_idx[u] for u in df_sub.index], dtype=np.int64)
 
    # --- warm-up numba (first call triggers compilation) ---
    if n >= 2:
        _ = compute_delta_swap(perm, L, 0, 1)
        _ = compute_delta_insert(perm, L, 0, 1)
        _ = compute_objective(perm[:2], L[:2, :2])
        _tmp = perm[:2].copy()
        _apply_insert(_tmp, 0, 1)
        _ = compute_lower_bound(L[:2, :2])
 
    # --- initial objective ---
    F = compute_objective(perm, L)
    init_F = F
    best_F = F
    best_perm = perm.copy()
 
    if verbose:
        print(f"[SA] n={n}, m={m}")
        print(f"[SA] Initial objective: {F}")
 
    if n <= 1:
        return df_sub.copy()
 
    # --- temperature schedule ---
    T0 = estimate_initial_temperature(perm, L, n_samples=min(1000, 50 * n), rng=rng)
    T_end = T0 * t_end_ratio
    L_markov = markov_factor * n
    total_stages = max(1, int(math.ceil(math.log(t_end_ratio) / math.log(alpha))))
    log_interval = max(1, total_stages // 10)
 
    if verbose:
        print(f"[SA] Estimated T0: {T0:.4f}")
        print(f"[SA] Total stages: {total_stages}, Markov chain length: {L_markov}")
 
    # --- main loop ---
    T = T0
    stage = 0
    t_start = time.time()
    stages_since_improve = 0
 
    while T > T_end:
        accepted = 0
        improved_this_stage = False
 
        for _ in range(L_markov):
            if rng.random() < swap_ratio:
                # ---- swap ----
                p = rng.randint(0, n - 2)
                max_dist = n - 1 - p
                dist = min(int(rng.expovariate(3.0 / n)) + 1, max_dist)
                q = p + dist
                delta = compute_delta_swap(perm, L, p, q)
 
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    perm[p], perm[q] = perm[q], perm[p]
                    F += delta
                    accepted += 1
                    if F < best_F:
                        best_F = F
                        best_perm = perm.copy()
                        improved_this_stage = True
            else:
                # ---- extract-insert ----
                p = rng.randint(0, n - 1)
                offset = int(rng.expovariate(3.0 / n)) + 1
                if rng.random() < 0.5:
                    q = min(p + offset, n - 1)
                else:
                    q = max(p - offset, 0)
                if q == p:
                    q = min(p + 1, n - 1)
 
                delta = compute_delta_insert(perm, L, p, q)
 
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    _apply_insert(perm, p, q)
                    F += delta
                    accepted += 1
                    if F < best_F:
                        best_F = F
                        best_perm = perm.copy()
                        improved_this_stage = True
 
        stage += 1
 
        # --- Reset-to-best if drifted too far without improvement ---
        if improved_this_stage:
            stages_since_improve = 0
        else:
            stages_since_improve += 1
 
        if stages_since_improve >= 10 and F > best_F * 1.02:
            perm[:] = best_perm
            F = best_F
            stages_since_improve = 0
 
        if verbose and (stage % log_interval == 0 or stage == 1):
            rate = accepted / L_markov if L_markov > 0 else 0.0
            elapsed = time.time() - t_start
            print(
                f"[SA] Stage {stage}/{total_stages} | T={T:.4f} | "
                f"F={F:.1f} | Best={best_F:.1f} | Accept={rate:.3f} | "
                f"{elapsed:.1f}s"
            )
 
        T *= alpha
 
    # --- final validation ---
    verified_F = compute_objective(best_perm, L)
    if verbose:
        improvement = (1 - best_F / init_F) * 100 if init_F != 0 else 0.0
        print(
            f"[SA] Finished. Best={best_F:.1f} | Verified={verified_F:.1f} | "
            f"Improvement: {improvement:.2f}% | {time.time() - t_start:.1f}s"
        )
 
    if abs(verified_F - best_F) > 0.5:
        print(
            f"[SA] WARNING: incremental F ({best_F}) != verified F ({verified_F}). "
            "Possible delta bug!"
        )
 
    # --- rebuild DataFrame ---
    reordered_unis = [unis[i] for i in best_perm]
    return df_sub.loc[reordered_unis]


# ---------------------------------------------------------------------------
# Multi-run wrapper (optional)
# ---------------------------------------------------------------------------
 
def simulated_annealing_multi_run(
    df_sub: pd.DataFrame, n_runs: int = 3, verbose: bool = True, **kwargs
) -> pd.DataFrame:
    """Run SA multiple times, each starting from the previous best."""
    best_result = None
    best_score = float("inf")
    current_input = df_sub  # 第一轮用原始输入

    for run in range(n_runs):
        if verbose:
            print(f"\n{'='*50}")
            print(f"[SA-Multi] Run {run + 1}/{n_runs}")
            print(f"{'='*50}")

        result = simulated_annealing(current_input, seed=run * 42 + 7, verbose=verbose, **kwargs)

        # score the result
        L, unis = build_inversion_matrix(result)
        uni_to_idx = {u: i for i, u in enumerate(unis)}
        perm = np.array([uni_to_idx[u] for u in result.index], dtype=np.int64)
        score = compute_objective(perm, L)

        if verbose:
            print(f"[SA-Multi] Run {run + 1} score: {score:.1f}")

        if score < best_score:
            best_score = score
            best_result = result

        # 下一轮从当前最优结果开始
        current_input = best_result

    if verbose:
        print(f"\n[SA-Multi] Best score across {n_runs} runs: {best_score:.1f}")

    return best_result