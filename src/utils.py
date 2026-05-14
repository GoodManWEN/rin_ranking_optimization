import pandas as pd
import numpy as np
from numba import njit


def load_rank_data(csv_path: str, m: int) -> pd.DataFrame:
    """读取CSV，提取 rank_1 ~ rank_m 列，index为大学标签。"""
    df = pd.read_csv(csv_path, index_col=0)
    rank_cols = [f"rank_{i}" for i in range(1, m + 1)]
    return df[rank_cols]

def extract_subset(df_rank: pd.DataFrame, subset: list) -> pd.DataFrame:
    """从排名DataFrame中抽取子集大学。"""
    return df_rank.loc[subset].copy()

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


def build_inversion_matrix_from_ranks(ranks):
    n, m = ranks.shape
    L = np.zeros((n, n), dtype=np.float64)
    for j in range(m):
        r = ranks[:, j]
        valid = ~np.isnan(r)
        both = valid[:, None] & valid[None, :]
        behind = r[:, None] > r[None, :]
        miss_vs_present = (~valid[:, None]) & valid[None, :]
        tied = both & (r[:, None] == r[None, :])
        L += (both & behind).astype(np.float64)
        L += miss_vs_present.astype(np.float64)
        L += 0.5 * tied.astype(np.float64)
    np.fill_diagonal(L, 0)
    return L


# ---------------------------------------------------------------------------
# Full objective (O(n²) – only for init / validation)
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
# Theoretical lower bound
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

# ======================================================================
# 工具函数
# ======================================================================
 
def evaluate(df_sub: pd.DataFrame) -> dict:
    """
    对一个排好序的 DataFrame 计算目标函数 F 和 lower bound gap。
    返回 {F, lower_bound, gap_pct}。
    """
    L, unis = build_inversion_matrix(df_sub)
    uni_to_idx = {u: i for i, u in enumerate(unis)}
    perm = np.array([uni_to_idx[u] for u in df_sub.index], dtype=np.int64)
 
    F = compute_objective(perm, L)
    lb = compute_lower_bound(L)
    gap = (F - lb) / lb * 100 if lb > 0 else 0.0
    return {"F": F, "lower_bound": lb, "gap_pct": gap}
 
 
def random_order(df_sub: pd.DataFrame, seed=0) -> pd.DataFrame:
    """将 DataFrame 的行随机打乱，作为无优化基线。"""
    rng = np.random.RandomState(seed)
    idx = list(df_sub.index)
    rng.shuffle(idx)
    return df_sub.loc[idx]