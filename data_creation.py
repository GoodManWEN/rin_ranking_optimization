#!/usr/bin/env python3
"""
University Ranking Pseudo-Data Generator
=========================================
Generates N universities with latent quality, 4 dimension scores, and M
independent ranking lists with distinct weighting profiles.

Generative model (3 layers):
  1. Latent quality  z_i ~ Beta(2, 5) ^ 0.75
  2. Dimension scores = factor_loading * z_i + correlated heteroscedastic noise
  3. Rankings = weighted_sum(dims, subjective_score) + perturbation

Output:
  - university_rankings.csv  : all scores and ranks
  - ranking_weights.csv      : weight profiles per ranking (ground truth)
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════
# Utility: Power Stretch
# ═══════════════════════════════════════════════════════════════════

def power_stretch(x: np.ndarray, gamma: float = 0.75) -> np.ndarray:
    """
    x^gamma (gamma < 1) 是一个凹变换。
    作用：拉开顶部学校的差距。Beta(2,5) 原始分布在顶部过于密集，
    经过 x^0.75 后，top-50 学校之间的间距增大约 30%，
    而底部学校被轻微压缩——这不影响排名，因为底部本来就难以区分。
    """
    return np.power(x, gamma)


# ═══════════════════════════════════════════════════════════════════
# Utility: Heteroscedastic Scale
# ═══════════════════════════════════════════════════════════════════

def hetero_scale(z: np.ndarray, floor: float = 0.30, strength: float = 0.70) -> np.ndarray:
    """
    返回一个 [floor, 1.0] 范围的缩放因子:
      scale(z) = 1 - strength * z

    z ≈ 1 (顶尖学校) → scale ≈ 0.30  → 噪声极小，各维度表现稳定
    z ≈ 0 (弱校)     → scale ≈ 1.00  → 噪声最大，出现"偏科"可能
    z ≈ 0.5 (中等)   → scale ≈ 0.65  → 中等波动

    依据：QS/THE 实际数据中，top-20 学校的排名年度波动 std ≈ 3-5 位，
    而 200-500 名学校的波动 std ≈ 50-100 位。这个 3-5 倍的差距主要
    由维度得分的方差差异驱动。
    """
    return np.clip(1 - strength * z, floor, 1.0)


# ═══════════════════════════════════════════════════════════════════
# Core: Dimension Score Generator
# ═══════════════════════════════════════════════════════════════════

def generate_dimension_scores(z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    从潜在质量 z 生成 4 个维度得分: research, teaching, international, reputation

    生成模型:
      d_{i,j} = λ_j · z_i  +  scale(z_i) · ε_{i,j}  +  special_effects_{i,j}

    其中 ε ~ MVN(0, Σ_noise)，维度间残差有弱相关性。

    Returns: (N, 4) array, 值域 [0, 100]
    """
    N = len(z)
    D = 4  # research, teaching, international, reputation

    # ── Factor loadings ──
    # 每个维度对 z 的响应强度不同
    # Research 和 Teaching 与综合实力高度相关 (0.85, 0.78)
    # International 相对独立 (0.50)，因为受地理/国家因素影响大
    # Reputation 中等 (0.72)，因为有声誉惯性和主观性
    factor_loadings = np.array([0.85, 0.78, 0.50, 0.72])

    # ── Residual noise covariance ──
    # 去除 z 的影响后，维度间仍存在的相关性
    # Teaching-Reputation: 0.35 (两者都含主观评价成分)
    # Research-Teaching: 0.25 (科研强的学校教学资源也好)
    # Research-Reputation: 0.20 (论文产出影响声誉)
    # International 与其他维度残差相关极低 (0.05-0.15)
    noise_std_base = np.array([0.10, 0.12, 0.18, 0.14])
    noise_corr = np.array([
        [1.00, 0.25, 0.08, 0.20],
        [0.25, 1.00, 0.15, 0.35],
        [0.08, 0.15, 1.00, 0.05],
        [0.20, 0.35, 0.05, 1.00],
    ])
    noise_cov = np.outer(noise_std_base, noise_std_base) * noise_corr

    # 生成多元正态噪声并施加异方差
    epsilon = rng.multivariate_normal(np.zeros(D), noise_cov, size=N)
    scale = hetero_scale(z, floor=0.30, strength=0.70).reshape(-1, 1)
    epsilon *= scale

    # 组合: 因子分 + 噪声
    dims = factor_loadings * z.reshape(-1, 1) + epsilon

    # ── Special Effect 1: 国际化的地理因子 ──
    # 现实：~25-30% 的学校位于小型开放国家 (荷兰、瑞士、新加坡、北欧等)
    # 这些学校国际化程度天然很高，不依赖于综合实力
    # 模型：二项抽样决定是否有地理加成，加成强度均匀分布
    geo_flag = rng.binomial(1, 0.28, size=N)
    geo_boost = rng.uniform(0.08, 0.22, size=N) * geo_flag
    dims[:, 2] += geo_boost

    # ── Special Effect 2: 声誉惯性 (prestige stickiness) ──
    # 现实：历史名校即使近期表现下滑，声誉仍居高不下
    # 模型：潜在质量 top-15% 的学校获得小幅声誉加成
    # 加成量 [0.03, 0.08] 相对于 [0,1] 尺度是温和的
    prestige_cutoff = np.percentile(z, 85)
    is_prestigious = z >= prestige_cutoff
    prestige_bonus = rng.uniform(0.03, 0.08, size=N) * is_prestigious
    dims[:, 3] += prestige_bonus

    # ── Special Effect 3: 科研维度的规模偏差 ──
    # 现实：大型综合大学 (论文总量高) 在 ARWU 类排名中有优势
    # 模型：~20% 中上水平学校 (z: 0.4-0.7) 获得科研小幅加成
    # 模拟"规模大但不顶尖"的学校
    size_candidates = (z >= 0.35) & (z <= 0.70)
    size_boost_mask = size_candidates & (rng.random(N) < 0.25)
    size_boost = rng.uniform(0.04, 0.10, size=N) * size_boost_mask
    dims[:, 0] += size_boost

    # 裁剪到 [0, 1] 并缩放到百分制
    dims = np.clip(dims, 0.0, 1.0)
    dims *= 100.0

    return dims


# ═══════════════════════════════════════════════════════════════════
# Core: Weight Profile Generator
# ═══════════════════════════════════════════════════════════════════

def generate_weight_profiles(M: int, rng: np.random.Generator) -> np.ndarray:
    """
    生成 M 个榜单的权重向量，每个长度为 5:
      [w_research, w_teaching, w_international, w_reputation, w_subjective]

    策略:
      - 前 4 个榜单使用预定义原型 (ARWU / THE / QS / 教学导向)
      - 第 5~M 个榜单：从两个随机原型的凸组合出发，施加 Dirichlet 扰动

    为什么用 Dirichlet 做扰动而不是简单加高斯噪声？
    因为权重必须非负且和为 1，Dirichlet 是 simplex 上的自然分布，
    浓度参数控制偏离程度。
    """
    # 4 个原型，反映现实排名体系的侧重差异
    # 每个原型的"主观/调研评分"权重统一占 20-30%，符合实际
    archetypes = np.array([
        # research, teaching, international, reputation, subjective
        [0.45, 0.15, 0.05, 0.10, 0.25],   # ARWU-like: 科研至上
        [0.25, 0.25, 0.10, 0.15, 0.25],   # THE-like:  综合均衡
        [0.10, 0.10, 0.15, 0.40, 0.25],   # QS-like:   声誉驱动
        [0.15, 0.35, 0.10, 0.15, 0.25],   # 教学导向型
    ])
    n_archetypes = len(archetypes)

    weights = np.zeros((M, 5))
    for k in range(M):
        if k < n_archetypes:
            # 原型榜单：基础权重 + 极小 Dirichlet 抖动
            # concentration=80 → 标准差约 0.01，几乎不偏离原型
            base = archetypes[k]
            jitter = rng.dirichlet(base * 80 + 1)  # 避免 alpha=0
            weights[k] = 0.90 * base + 0.10 * jitter
        else:
            # 衍生榜单：两原型的随机凸组合 + 较大 Dirichlet 抖动
            idx1, idx2 = rng.choice(n_archetypes, size=2, replace=False)
            alpha = rng.uniform(0.25, 0.75)
            base = alpha * archetypes[idx1] + (1 - alpha) * archetypes[idx2]
            jitter = rng.dirichlet(base * 30 + 1)
            weights[k] = 0.75 * base + 0.25 * jitter

        # 归一化，确保权重和为 1
        weights[k] = np.maximum(weights[k], 0.01)  # 最低 1% 权重
        weights[k] /= weights[k].sum()

    return weights


# ═══════════════════════════════════════════════════════════════════
# Core: Subjective Score Generator
# ═══════════════════════════════════════════════════════════════════

def generate_subjective_score(
    z: np.ndarray,
    rng: np.random.Generator,
    base_noise_std: float = 0.12,
    ranker_bias_std: float = 0.03,
) -> np.ndarray:
    """
    生成单个榜单的主观评分分量。

    模型:
      s_i = z_i + bias_k + noise_i

    其中:
      - bias_k ~ N(0, 0.03): 榜单层面的系统性偏差
        (有的评审团整体偏乐观/悲观)
      - noise_i ~ N(0, σ(z_i)): 个体层面噪声，异方差
        σ(z_i) = base_std * (1 - 0.5*z_i)
        顶尖学校的主观评价一致性更高 (评审共识度高)

    为什么主观分也用异方差？
      因为 Harvard/MIT 这种学校，任何评审团都会给高分 (共识强)，
      但排名 300-500 的学校，不同评审团的评价可能相差很大。

    Returns: (N,) array, 值域 [0, 100]
    """
    N = len(z)

    # 榜单级系统偏差
    bias = rng.normal(0, ranker_bias_std)

    # 个体级异方差噪声
    individual_std = base_noise_std * (1 - 0.5 * z)
    noise = rng.normal(0, individual_std)

    subj = z + bias + noise
    return np.clip(subj, 0, 1) * 100


# ═══════════════════════════════════════════════════════════════════
# Core: Final Score + Ranking
# ═══════════════════════════════════════════════════════════════════

def compute_ranking(
    dimensions: np.ndarray,
    z: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    对每个榜单计算最终分数并排名。

    最终得分 = Σ w_j * dim_j + w_subj * subjective + perturbation

    perturbation 模拟数据采集误差、指标计算口径差异等:
      η_i ~ N(0, σ_η(z_i))
      σ_η = 1.5 * (1 - 0.5*z_i)  → 大约在 0.75~1.5 分 (百分制) 范围

    Returns:
      raw_scores: (N, M) 原始得分
      rankings:   (N, M) 排名 (1-indexed)
    """
    N, M = len(z), weights.shape[0]
    raw_scores = np.zeros((N, M))
    rankings = np.zeros((N, M), dtype=int)

    for k in range(M):
        w = weights[k]

        # 主观分 (每个榜单独立生成)
        subj = generate_subjective_score(z, rng)

        # 加权求和
        score = (
            w[0] * dimensions[:, 0] +
            w[1] * dimensions[:, 1] +
            w[2] * dimensions[:, 2] +
            w[3] * dimensions[:, 3] +
            w[4] * subj
        )

        # 最终微扰动
        perturb_std = 1.5 * hetero_scale(z, floor=0.50, strength=0.50)
        score += rng.normal(0, perturb_std)

        raw_scores[:, k] = score
        # argsort(-score) 给出降序索引，再 argsort 得到排名
        rankings[:, k] = np.argsort(-score).argsort() + 1

    return raw_scores, rankings


# ═══════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════

def generate_rankings(N: int = 2000, M: int = 10, seed: int = 42, output_dir: str = './output'):
    """
    生成伪高校排名数据。

    Args:
        N: 高校数量
        M: 榜单数量 (建议 >= 4，因为有 4 个权重原型)
        seed: 随机种子
        output_dir: 输出目录

    Outputs:
        university_rankings.csv:
            - university_id: 学校编号 U0001..U{N}
            - latent_quality: 潜在质量 (0-100)，用于验证聚合算法
            - dim_research/teaching/international/reputation: 维度得分 (0-100)
            - rank_1..rank_M: 各榜单排名
            - score_1..score_M: 各榜单原始得分

        ranking_weights.csv:
            - 各榜单的权重配置 (ground truth)
    """
    rng = np.random.default_rng(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    DIM_NAMES = ['research', 'teaching', 'international', 'reputation']

    # ── Layer 1: Latent Quality ──
    z_raw = rng.beta(2, 5, size=N)
    z = power_stretch(z_raw, gamma=0.75)

    # ── Layer 2: Dimension Scores ──
    dims = generate_dimension_scores(z, rng)

    # ── Layer 3: Weight Profiles ──
    weights = generate_weight_profiles(M, rng)

    # ── Layer 4: Scoring & Ranking ──
    raw_scores, rankings = compute_ranking(dims, z, weights, rng)

    # ── Assemble & Export ──
    data = {'university_id': [f'U{i+1:04d}' for i in range(N)]}
    data['latent_quality'] = np.round(z * 100, 2)
    for j, name in enumerate(DIM_NAMES):
        data[f'dim_{name}'] = np.round(dims[:, j], 2)
    for k in range(M):
        data[f'rank_{k+1}'] = rankings[:, k]
    for k in range(M):
        data[f'score_{k+1}'] = np.round(raw_scores[:, k], 2)

    df = pd.DataFrame(data)
    df.to_csv(out / 'university_rankings.csv', index=False)

    weight_rows = []
    for k in range(M):
        weight_rows.append({
            'ranking_id': k + 1,
            'w_research': round(weights[k, 0], 4),
            'w_teaching': round(weights[k, 1], 4),
            'w_international': round(weights[k, 2], 4),
            'w_reputation': round(weights[k, 3], 4),
            'w_subjective': round(weights[k, 4], 4),
        })
    df_weights = pd.DataFrame(weight_rows)
    df_weights.to_csv(out / 'ranking_weights.csv', index=False)

    return df, df_weights


# ═══════════════════════════════════════════════════════════════════
# Sanity Check (run as script)
# ═══════════════════════════════════════════════════════════════════

def sanity_check(df: pd.DataFrame, df_weights: pd.DataFrame):
    """打印关键统计量，快速验证数据质量。"""
    M = len(df_weights)
    rank_cols = [f'rank_{k+1}' for k in range(M)]

    print("=" * 60)
    print("SANITY CHECK")
    print("=" * 60)

    # 1. 潜在质量分布
    lq = df['latent_quality']
    print(f"\n[Latent Quality Distribution]")
    print(f"  Mean={lq.mean():.1f}  Median={lq.median():.1f}  "
          f"Std={lq.std():.1f}  Min={lq.min():.1f}  Max={lq.max():.1f}")

    # 2. 维度得分相关矩阵 (含 latent_quality)
    dim_cols = ['latent_quality', 'dim_research', 'dim_teaching',
                'dim_international', 'dim_reputation']
    corr = df[dim_cols].corr()
    print(f"\n[Dimension Correlation Matrix]")
    print(corr.round(3).to_string())

    # 3. 排名波动: 不同水平学校的跨榜单排名标准差
    rank_std = df[rank_cols].std(axis=1)
    # 按 latent_quality 分层
    bins = [0, 10, 25, 50, 75, 100]
    labels = ['top-10%', '10-25%', '25-50%', '50-75%', 'bottom-25%']
    # 百分位是按 latent_quality 降序的
    pcts = df['latent_quality'].rank(pct=True, ascending=False) * 100
    tier = pd.cut(pcts, bins=bins, labels=labels, right=True)
    tier_stats = rank_std.groupby(tier).agg(['mean', 'median', 'min', 'max'])
    print(f"\n[Rank Spread by Quality Tier (std across {M} rankings)]")
    print(tier_stats.round(1).to_string())

    # 4. 榜单间排名相关性
    rank_corr = df[rank_cols].corr(method='spearman').values
    triu = rank_corr[np.triu_indices_from(rank_corr, k=1)]
    print(f"\n[Pairwise Spearman Rank Correlation]")
    print(f"  Min={triu.min():.3f}  Max={triu.max():.3f}  Mean={triu.mean():.3f}")

    # 5. 顶尖学校在各榜单的排名范围
    top_by_lq = df.nlargest(20, 'latent_quality')
    top_rank_min = top_by_lq[rank_cols].min(axis=1)
    top_rank_max = top_by_lq[rank_cols].max(axis=1)
    top_rank_range = top_rank_max - top_rank_min
    print(f"\n[Top-20 Schools (by latent quality): Rank Range Across Rankings]")
    print(f"  Mean range={top_rank_range.mean():.0f}  "
          f"Max range={top_rank_range.max():.0f}  "
          f"Median best rank={top_rank_min.median():.0f}")

    # 6. 权重配置
    print(f"\n[Weight Profiles]")
    print(df_weights.to_string(index=False))

    print("\n" + "=" * 60)


if __name__ == '__main__':
    import sys

    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    M = int(sys.argv[2]) if len(sys.argv) > 2 else 11

    print(f"Generating {N} universities × {M} rankings...")
    df, df_w = generate_rankings(N=N, M=M, seed=42, output_dir='./output')
    sanity_check(df, df_w)
    print(f"\nFiles saved to ./output/")