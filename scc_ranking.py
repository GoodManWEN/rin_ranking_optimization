"""
SCC-based university ranking grouping.
Input:  CSV file + number of rankings m.
Output: Print groups from strongest to weakest via SCC + topological sort.
"""
 
import sys
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from collections import defaultdict, deque
import math
import random
import time
from numba import njit

 
def scc_grouping(csv_path, m) -> defaultdict:
    # 1. 读取数据, 只提取 rank_1 ~ rank_m
    df = pd.read_csv(csv_path, index_col=0)
    rank_cols = [f"rank_{i}" for i in range(1, m + 1)]
    df = df[rank_cols]
 
    universities = df.index.tolist()
    n = len(universities)
    ranks = df.values.astype(float)  # n x m, NaN for missing
    threshold = m // 2 + 1           # 严格多数: 11 -> 6
 
    print(f"Universities: {n}, Rankings: {m}, Majority threshold: {threshold}")
 
    # 2. 构建有向图: 多数投票
    win_count = np.zeros((n, n), dtype=np.int32)
 
    for j in range(m):
        r = ranks[:, j]
        valid = ~np.isnan(r)
        both_valid = valid[:, None] & valid[None, :]
        a_valid_b_invalid = valid[:, None] & ~valid[None, :]
        a_better = r[:, None] < r[None, :]
        win_count += (both_valid & a_better | a_valid_b_invalid).astype(np.int32)
 
    adj = (win_count >= threshold).astype(np.int8)
    np.fill_diagonal(adj, 0)
    print(f"Directed edges: {int(adj.sum())}")
 
    # 3. 求强连通分量
    graph = csr_matrix(adj)
    n_comp, labels = connected_components(graph, directed=True, connection='strong')
    print(f"SCC components: {n_comp}")
 
    # 4. 缩点 + 拓扑排序 (Kahn's algorithm)
    rows, cols = graph.nonzero()
    scc_edges = set()
    for a, b in zip(rows, cols):
        if labels[a] != labels[b]:
            scc_edges.add((labels[a], labels[b]))
 
    dag = defaultdict(set)
    in_deg = [0] * n_comp
    for u, v in scc_edges:
        dag[u].add(v)
        in_deg[v] += 1
 
    queue = deque(i for i in range(n_comp) if in_deg[i] == 0)
    topo = []
    while queue:
        node = queue.popleft()
        topo.append(node)
        for nb in dag[node]:
            in_deg[nb] -= 1
            if in_deg[nb] == 0:
                queue.append(nb)
 
    # 5. 按拓扑序输出: 排名最高的组最先打印
    groups = defaultdict(list)
    for i, lbl in enumerate(labels):
        groups[lbl].append(universities[i])
 

    # print(f"\n{'='*60}")
    # print(f"Results: {n} universities -> {n_comp} groups (strongest first)")
    # print(f"{'='*60}")
    # for rank, scc_id in enumerate(topo, 1):
    #     members = groups[scc_id]
        # print(f"\nGroup {rank} ({len(members)} universities): {members}")
 
    return topo, groups, labels


def load_rank_data(csv_path: str, m: int) -> pd.DataFrame:
    """读取CSV，提取 rank_1 ~ rank_m 列，index为大学标签。"""
    df = pd.read_csv(csv_path, index_col=0)
    rank_cols = [f"rank_{i}" for i in range(1, m + 1)]
    return df[rank_cols]
 
 
def extract_subset(df_rank: pd.DataFrame, subset: list) -> pd.DataFrame:
    """从排名DataFrame中抽取子集大学。"""
    return df_rank.loc[subset].copy()
 
 
def borda_sort(df_sub: pd.DataFrame) -> pd.DataFrame:
    """
    对子集DataFrame执行Borda计分并排序。
    每列(榜单)内：非NaN值按排名从优到劣赋分 n-1, n-2, ..., 0；NaN赋0。
    最终按总分降序排列。
    """
    borda_scores = pd.DataFrame(0.0, index=df_sub.index, columns=df_sub.columns)
 
    for col in df_sub.columns:
        valid = df_sub[col].dropna().sort_values()
        k = len(valid)
        for rank_pos, uni in enumerate(valid.index):
            borda_scores.at[uni, col] = k - 1 - rank_pos
 
    result = df_sub.copy()
    result["borda_total"] = borda_scores.sum(axis=1)
    result = result.sort_values("borda_total", ascending=False)
    result = result.drop(columns=["borda_total"])
    return result
 
def majority_compare(a: str, b: str, df_sub: pd.DataFrame) -> bool:
    """
    判断A是否优于B（排名更高）。
    仅统计双方都有排名的榜单，若多数认为A排名更小(更优)则返回True。
    平局时返回False（保持原序）。
    """
    a_ranks = df_sub.loc[a]
    b_ranks = df_sub.loc[b]
    valid = a_ranks.notna() & b_ranks.notna()
    return (a_ranks[valid] < b_ranks[valid]).sum() > valid.sum() / 2

def insertion_sort_by_majority(df_sub: pd.DataFrame) -> pd.DataFrame:
    """
    基于多数投票的插入排序。
    输入为已经Borda排序的df_sub（仅含rank列，不含borda_total）。
    输出按多数比较优化后的排序结果。
    """
    # 仅使用rank列做比较
    rank_cols = [c for c in df_sub.columns if c.startswith("rank_")]
    df_rank_only = df_sub[rank_cols]
 
    unis = list(df_rank_only.index)
 
    for i in range(1, len(unis)):
        key = unis[i]
        j = i - 1
        while j >= 0 and majority_compare(key, unis[j], df_rank_only):
            unis[j + 1] = unis[j]
            j -= 1
        unis[j + 1] = key
 
    return df_sub.loc[unis]



# ---------------------------------------------------------------------------
# 1. Inversion matrix
# ---------------------------------------------------------------------------
 
def build_inversion_matrix(df_sub: pd.DataFrame):
    """
    Build the inversion matrix L from ranking data.
 
    Parameters
    ----------
    df_sub : pd.DataFrame
        DataFrame with rank_1 … rank_m columns; index = university labels.
 
    Returns
    -------
    L : np.ndarray, shape (n, n)
        L[i][j] = number of ranking systems where university i is ranked
        *behind* university j.
    unis : list
        University labels aligned with L's row/column indices.
    """
    unis = list(df_sub.index)
    n = len(unis)
    ranks = df_sub.values.astype(float)  # n × m
 
    L = np.zeros((n, n), dtype=np.float64)
    for j in range(ranks.shape[1]):
        r = ranks[:, j]
        valid = ~np.isnan(r)
        both_valid = valid[:, None] & valid[None, :]
        i_behind_j = r[:, None] > r[None, :]
        i_invalid_j_valid = (~valid[:, None]) & valid[None, :]
        tied = both_valid & (r[:, None] == r[None, :])
 
        L += (both_valid & i_behind_j).astype(np.float64)
        L += i_invalid_j_valid.astype(np.float64)
        L += 0.5 * tied.astype(np.float64)
 
    np.fill_diagonal(L, 0)
    return L, unis
 
 
# ---------------------------------------------------------------------------
# 2. Full objective (O(n²) – only for init / validation)
# ---------------------------------------------------------------------------
 
@njit
def compute_objective(perm, L):
    """Sum of L[perm[p], perm[q]] for all p < q."""
    n = perm.shape[0]
    F = 0.0
    for p in range(n):
        ip = perm[p]
        for q in range(p + 1, n):
            F += L[ip, perm[q]]
    return F
 
 
# ---------------------------------------------------------------------------
# 3. Incremental delta calculations  (hot path – numba-accelerated)
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
# 4. Permutation mutation helpers  (numpy, in-place)
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
# 5. Initial temperature estimation  (FIXED: short-range + conservative)
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
 
 
# ---------------------------------------------------------------------------
# 6. Theoretical lower bound
# ---------------------------------------------------------------------------
 
@njit
def compute_lower_bound(L):
    """Sum of min(L[i,j], L[j,i]) for all i < j."""
    n = L.shape[0]
    lb = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = L[i, j], L[j, i]
            lb += a if a < b else b
    return lb
 
 
# ---------------------------------------------------------------------------
# 7. Main SA routine
# ---------------------------------------------------------------------------
 
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
# 8. Multi-run wrapper (optional)
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

 
if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "./output/university_rankings.csv"
    m = int(sys.argv[2]) if len(sys.argv) > 2 else 11
    _, groups, _ = scc_grouping(csv_path, m)

    df_rank_all = load_rank_data(csv_path, m)
    for scc_id in sorted(groups.keys()):
        if len(groups[scc_id]) <= 1: # 此处待修改
            continue 

        df_sub = extract_subset(df_rank_all, groups[scc_id])
        df_borda_sorted = borda_sort(df_sub)
        df_insertion_sorted = insertion_sort_by_majority(df_borda_sorted)
        

        if len(df_insertion_sorted) > 50:
            df_annealing_sorted = simulated_annealing_multi_run(df_insertion_sorted, verbose=True, n_runs=3)
        else:
            df_annealing_sorted = df_insertion_sorted

        print(df_annealing_sorted)