import numpy as np
import time
from numba import njit
from .utils import build_inversion_matrix_from_ranks


# ======================================================================
# Numba-accelerated core routines
# ======================================================================

@njit
def _compute_objective(perm, L):
    n = perm.shape[0]
    F = 0.0
    for p in range(n):
        for q in range(p + 1, n):
            F += L[perm[p], perm[q]]
    return F

@njit
def _bnb_recurse(sub_L, k, perm, used, depth, partial_cost,
                 best_cost, best_perm):
    """Branch-and-bound exhaustive search over permutations of k elements."""
    if depth == k:
        if partial_cost < best_cost[0]:
            best_cost[0] = partial_cost
            for i in range(k):
                best_perm[i] = perm[i]
        return

    for elem in range(k):
        if used[elem]:
            continue

        # Cost of placing elem at position depth:
        # sum of sub_L[prev_element, elem] for all elements already placed
        add_cost = 0.0
        for d in range(depth):
            add_cost += sub_L[perm[d], elem]

        new_cost = partial_cost + add_cost

        # Prune: if partial cost already >= best, skip
        if new_cost >= best_cost[0]:
            continue

        perm[depth] = elem
        used[elem] = True
        _bnb_recurse(sub_L, k, perm, used, depth + 1, new_cost,
                     best_cost, best_perm)
        used[elem] = False

    return

@njit
def _exhaustive_search(sub_L, k):
    """
    Find the permutation of 0..k-1 minimising sum of sub_L[perm[i], perm[j]]
    for i < j.  Uses branch-and-bound with pruning.
    Returns (best_cost, best_perm).
    """
    perm = np.zeros(k, dtype=np.int64)
    used = np.zeros(k, dtype=np.bool_)
    best_perm = np.arange(k, dtype=np.int64)
    best_cost = np.array([1e18], dtype=np.float64)

    # Compute initial upper bound from identity permutation
    ub = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            ub += sub_L[i, j]
    best_cost[0] = ub

    _bnb_recurse(sub_L, k, perm, used, 0, 0.0, best_cost, best_perm)
    return best_cost[0], best_perm


@njit
def _insertion_sort_by_L(sub_L, k, threshold):
    """Majority-based insertion sort: O(k^2), fallback for large segments."""
    order = np.arange(k, dtype=np.int64)
    for i in range(1, k):
        key = order[i]
        j = i - 1
        while j >= 0 and sub_L[order[j], key] > threshold:
            order[j + 1] = order[j]
            j -= 1
        order[j + 1] = key
    return order

# ======================================================================
# Natural breakpoint detection  (paper's logic)
# ======================================================================

def find_natural_breakpoints(L_local, m):
    """
    Position j is a breakpoint iff ALL L_local[k, j] <= m/2 for k > j.
    Returns sorted list; n-1 is always included.
    """
    n = L_local.shape[0]
    thr = m / 2.0
    bps = []
    for j in range(n - 1):
        if np.all(L_local[j + 1:, j] <= thr):
            bps.append(j)
    bps.append(n - 1)
    return bps


def breakpoints_to_segments(bps, n):
    segs = []
    prev = 0
    for bp in bps:
        segs.append((prev, bp + 1))
        prev = bp + 1
    return segs


# ======================================================================
# Paper-style optimise: breakpoints + segment exhaustive/fallback
# ======================================================================

def paper_style_optimise(L, perm, m, max_exhaust=10, max_iter=5, verbose=False):
    """
    Paper's algorithm applied to a sub-permutation.
    Iterate: find breakpoints -> optimise each segment -> repeat.
    """
    n = len(perm)
    if n <= 1:
        return perm.copy()

    perm = perm.copy()
    thr = m / 2.0

    for it in range(max_iter):
        L_local = L[np.ix_(perm, perm)]
        bps = find_natural_breakpoints(L_local, m)
        segs = breakpoints_to_segments(bps, n)

        if verbose:
            sizes = [e - s for s, e in segs]
            print(f"    [Paper] Iter {it+1}: {len(segs)} segs, "
                  f"max={max(sizes)}, sizes={sizes[:15]}{'...' if len(sizes)>15 else ''}")

        changed = False
        new_perm = perm.copy()

        for s, e in segs:
            k = e - s
            if k <= 1:
                continue

            seg_idx = perm[s:e].copy()
            sub_L = L[np.ix_(seg_idx, seg_idx)]

            if k <= max_exhaust:
                _, best_local = _exhaustive_search(sub_L, k)
                optimised = seg_idx[best_local]
            else:
                order = _insertion_sort_by_L(sub_L, k, thr)
                optimised = seg_idx[order]

            if not np.array_equal(optimised, seg_idx):
                changed = True
            new_perm[s:e] = optimised

        perm = new_perm
        if not changed:
            if verbose:
                print(f"    [Paper] Converged at iteration {it+1}")
            break

    return perm



# ======================================================================
# Sliding window refinement
# ======================================================================

def sliding_window_refinement(L, perm, m,
                              window_size=20, stride=None,
                              max_exhaust=10, max_rounds=15,
                              verbose=True):
    if stride is None:
        stride = window_size // 2

    n = len(perm)
    if n <= window_size:
        return paper_style_optimise(L, perm, m, max_exhaust, verbose=verbose)

    perm = perm.copy()
    F_before = _compute_objective(perm, L)

    if verbose:
        print(f"\n[SW] n={n}, window={window_size}, stride={stride}")
        print(f"[SW] F before: {F_before:.1f}")

    t0 = time.time()

    for rd in range(max_rounds):
        improvements = 0
        n_win = 0

        # Forward pass
        pos = 0
        while pos < n:
            end = min(pos + window_size, n)
            if end - pos < 3:
                break
            win = perm[pos:end].copy()
            opt = paper_style_optimise(L, win, m, max_exhaust)
            if not np.array_equal(opt, win):
                perm[pos:end] = opt
                improvements += 1
            n_win += 1
            pos += stride

        # Backward pass
        pos = max(0, n - window_size)
        while pos >= 0:
            end = min(pos + window_size, n)
            if end - pos < 3:
                break
            win = perm[pos:end].copy()
            opt = paper_style_optimise(L, win, m, max_exhaust)
            if not np.array_equal(opt, win):
                perm[pos:end] = opt
                improvements += 1
            n_win += 1
            pos -= stride
            if pos < 0:
                break

        F_now = _compute_objective(perm, L)
        if verbose:
            print(f"[SW] Round {rd+1}: {improvements}/{n_win} improved | "
                  f"F={F_now:.1f} | {time.time()-t0:.1f}s")

        if improvements == 0:
            if verbose:
                print(f"[SW] Converged at round {rd+1}")
            break

    F_final = _compute_objective(perm, L)
    if verbose:
        d = F_before - F_final
        pct = d / F_before * 100 if F_before > 0 else 0
        print(f"[SW] Done: {F_before:.1f} -> {F_final:.1f} "
              f"(delta={d:.1f}, {pct:.3f}%) | {time.time()-t0:.1f}s")
    return perm



def refine_after_sa(df_sub, window_size=20, max_exhaust=10,
                    max_rounds=15, verbose=True):
    """
    Drop-in refinement after simulated_annealing().

    Usage:
        df_sa = simulated_annealing_multi_run(df_insertion_sorted, ...)
        df_refined = refine_after_sa(df_sa, window_size=20)
    """
    unis = list(df_sub.index)
    n = len(unis)
    m = len(df_sub.columns)
    ranks = df_sub.values.astype(float)

    L = build_inversion_matrix_from_ranks(ranks)
    perm = np.arange(n, dtype=np.int64)

    # Warm up numba
    if n >= 2:
        _compute_objective(perm[:2], L[:2, :2])
        _sub = L[:2, :2].copy()
        _exhaustive_search(_sub, 2)
        _insertion_sort_by_L(_sub, 2, m / 2.0)

    # Lower bound
    lb = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            lb += min(L[i, j], L[j, i])

    F_init = _compute_objective(perm, L)
    gap_bef = (F_init - lb) / lb * 100 if lb > 0 else 0

    if verbose:
        print(f"\n{'='*60}")
        print(f"[Refine] n={n}, m={m}")
        print(f"[Refine] Lower bound: {lb:.1f}")
        print(f"[Refine] SA objective: {F_init:.1f} (gap: {gap_bef:.3f}%)")
        print(f"{'='*60}")

    ref = sliding_window_refinement(L, perm, m,
                                    window_size=window_size,
                                    max_exhaust=max_exhaust,
                                    max_rounds=max_rounds,
                                    verbose=verbose)

    F_fin = _compute_objective(ref, L)
    gap_aft = (F_fin - lb) / lb * 100 if lb > 0 else 0
    if verbose:
        print(f"[Refine] Final: {F_fin:.1f} (gap: {gap_aft:.3f}%)")

    reordered = [unis[i] for i in ref]
    return df_sub.loc[reordered]
    