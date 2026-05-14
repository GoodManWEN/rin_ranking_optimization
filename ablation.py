import sys
import time
import pandas as pd
from collections import OrderedDict
from utils import load_rank_data, extract_subset, random_order, evaluate
from insertion_sorting import insertion_sort_by_majority
from borda_sorting import borda_sort
from simulated_annealings import simulated_annealing
from sliding_window_rin import refine_after_sa
from scc_grouping import scc_grouping



# ======================================================================
# 实验 A：累积实验 —— 逐步叠加每个步骤
# ======================================================================
 
def run_cumulative(df_sub: pd.DataFrame, scc_label: str, m: int):
    """
    逐步叠加 pipeline 的每个步骤，记录每步的耗时和效果。
 
    Pipeline 顺序:
      Step 0: Random (baseline)
      Step 1: Borda
      Step 2: Borda → Insertion
      Step 3: Borda → Insertion → SA
      Step 4: Borda → Insertion → SA → SW  (full)
    """
    n = len(df_sub)
    results = OrderedDict()
 
    print(f"\n{'─'*70}")
    print(f"[Cumulative] SCC={scc_label}, n={n}, m={m}")
    print(f"{'─'*70}")
 
    # ── Step 0: Random baseline ──
    t0 = time.time()
    df_random = random_order(df_sub, seed=0)
    t_random = time.time() - t0
    ev = evaluate(df_random)
    results["0_Random"] = {**ev, "time": t_random, "step_time": t_random}
    print(f"  [Step 0] Random       | F={ev['F']:.1f} | gap={ev['gap_pct']:.3f}% | {t_random:.2f}s")
 
    # ── Step 1: Borda ──
    t0 = time.time()
    df_borda = borda_sort(df_sub)
    t_borda = time.time() - t0
    ev = evaluate(df_borda)
    results["1_Borda"] = {**ev, "time": t_borda, "step_time": t_borda}
    print(f"  [Step 1] Borda        | F={ev['F']:.1f} | gap={ev['gap_pct']:.3f}% | {t_borda:.2f}s")
 
    # ── Step 2: Borda → Insertion ──
    t0 = time.time()
    df_insert = insertion_sort_by_majority(df_borda)
    t_insert = time.time() - t0
    ev = evaluate(df_insert)
    results["2_Borda+Ins"] = {**ev, "time": t_borda + t_insert, "step_time": t_insert}
    print(f"  [Step 2] + Insertion   | F={ev['F']:.1f} | gap={ev['gap_pct']:.3f}% | {t_insert:.2f}s")
 
    # ── Step 3: Borda → Insertion → SA ──
    t0 = time.time()
    if n > 50:
        df_sa = simulated_annealing(df_insert, seed=0, verbose=False)
    else:
        df_sa = df_insert.copy()
    t_sa = time.time() - t0
    ev = evaluate(df_sa)
    results["3_Borda+Ins+SA"] = {**ev, "time": t_borda + t_insert + t_sa, "step_time": t_sa}
    print(f"  [Step 3] + SA          | F={ev['F']:.1f} | gap={ev['gap_pct']:.3f}% | {t_sa:.2f}s")
 
    # ── Step 4: Borda → Insertion → SA → SW (full) ──
    t0 = time.time()
    df_sw = refine_after_sa(df_sa, window_size=20, verbose=False)
    t_sw = time.time() - t0
    ev = evaluate(df_sw)
    results["4_Full"] = {**ev, "time": t_borda + t_insert + t_sa + t_sw, "step_time": t_sw}
    print(f"  [Step 4] + SW (full)   | F={ev['F']:.1f} | gap={ev['gap_pct']:.3f}% | {t_sw:.2f}s")
 
    return results
 
 
# ======================================================================
# 实验 B：Skip-one 实验 —— 从完整流水线中去掉一个步骤
# ======================================================================
 
def run_skip_one(df_sub: pd.DataFrame, scc_label: str, m: int):
    """
    从完整 pipeline 中每次去掉一个步骤，观察性能退化。
 
    Configs:
      Full:          Borda → Insertion → SA → SW
      Skip Borda:    Random → Insertion → SA → SW
      Skip Insert:   Borda → SA → SW
      Skip SA:       Borda → Insertion → SW
      Skip SW:       Borda → Insertion → SA
    """
    n = len(df_sub)
    results = OrderedDict()
 
    print(f"\n{'─'*70}")
    print(f"[Skip-One] SCC={scc_label}, n={n}, m={m}")
    print(f"{'─'*70}")
 
    # ── Full pipeline (reference) ──
    t0 = time.time()
    df_borda = borda_sort(df_sub)
    df_insert = insertion_sort_by_majority(df_borda)
    df_sa = simulated_annealing(df_insert, seed=0, verbose=False) if n > 50 else df_insert.copy()
    df_sw = refine_after_sa(df_sa, window_size=20, verbose=False)
    t_full = time.time() - t0
    ev = evaluate(df_sw)
    results["Full"] = {**ev, "time": t_full}
    F_full = ev["F"]
    print(f"  [Full]         | F={ev['F']:.1f} | gap={ev['gap_pct']:.3f}% | {t_full:.2f}s")
 
    # ── Skip Borda: Random → Insertion → SA → SW ──
    t0 = time.time()
    df_rand = random_order(df_sub, seed=0)
    df_ins2 = insertion_sort_by_majority(df_rand)
    df_sa2 = simulated_annealing(df_ins2, seed=0, verbose=False) if n > 50 else df_ins2.copy()
    df_sw2 = refine_after_sa(df_sa2, window_size=20, verbose=False)
    t_skip = time.time() - t0
    ev = evaluate(df_sw2)
    results["Skip_Borda"] = {**ev, "time": t_skip}
    print(f"  [Skip Borda]   | F={ev['F']:.1f} | gap={ev['gap_pct']:.3f}% | "
          f"delta_F=+{ev['F']-F_full:.1f} | {t_skip:.2f}s")
 
    # ── Skip Insertion: Borda → SA → SW ──
    t0 = time.time()
    df_borda3 = borda_sort(df_sub)
    df_sa3 = simulated_annealing(df_borda3, seed=0, verbose=False) if n > 50 else df_borda3.copy()
    df_sw3 = refine_after_sa(df_sa3, window_size=20, verbose=False)
    t_skip = time.time() - t0
    ev = evaluate(df_sw3)
    results["Skip_Insert"] = {**ev, "time": t_skip}
    print(f"  [Skip Insert]  | F={ev['F']:.1f} | gap={ev['gap_pct']:.3f}% | "
          f"delta_F=+{ev['F']-F_full:.1f} | {t_skip:.2f}s")
 
    # ── Skip SA: Borda → Insertion → SW ──
    t0 = time.time()
    df_borda4 = borda_sort(df_sub)
    df_ins4 = insertion_sort_by_majority(df_borda4)
    df_sw4 = refine_after_sa(df_ins4, window_size=20, verbose=False)
    t_skip = time.time() - t0
    ev = evaluate(df_sw4)
    results["Skip_SA"] = {**ev, "time": t_skip}
    print(f"  [Skip SA]      | F={ev['F']:.1f} | gap={ev['gap_pct']:.3f}% | "
          f"delta_F=+{ev['F']-F_full:.1f} | {t_skip:.2f}s")
 
    # ── Skip SW: Borda → Insertion → SA ──
    t0 = time.time()
    df_borda5 = borda_sort(df_sub)
    df_ins5 = insertion_sort_by_majority(df_borda5)
    df_sa5 = simulated_annealing(df_ins5, seed=0, verbose=False) if n > 50 else df_ins5.copy()
    t_skip = time.time() - t0
    ev = evaluate(df_sa5)
    results["Skip_SW"] = {**ev, "time": t_skip}
    print(f"  [Skip SW]      | F={ev['F']:.1f} | gap={ev['gap_pct']:.3f}% | "
          f"delta_F=+{ev['F']-F_full:.1f} | {t_skip:.2f}s")
 
    return results
 
 
# ======================================================================
# 实验 C：单步骤独立能力测试 —— 只用一个步骤能做到多好
# ======================================================================
 
def run_isolated(df_sub: pd.DataFrame, scc_label: str, m: int):
    """
    每个步骤单独运行（不依赖前序步骤），测试其独立优化能力。
 
    Configs:
      Borda only
      Insertion only (from random)
      SA only (from random)
      SW only (from random)
    """
    n = len(df_sub)
    results = OrderedDict()
 
    print(f"\n{'─'*70}")
    print(f"[Isolated] SCC={scc_label}, n={n}, m={m}")
    print(f"{'─'*70}")
 
    # ── Random baseline ──
    df_rand = random_order(df_sub, seed=0)
    ev_rand = evaluate(df_rand)
    results["Random"] = ev_rand
    print(f"  [Random]          | F={ev_rand['F']:.1f} | gap={ev_rand['gap_pct']:.3f}%")
 
    # ── Borda only ──
    t0 = time.time()
    df_b = borda_sort(df_sub)
    t_elapsed = time.time() - t0
    ev = evaluate(df_b)
    results["Borda_only"] = {**ev, "time": t_elapsed}
    print(f"  [Borda only]      | F={ev['F']:.1f} | gap={ev['gap_pct']:.3f}% | {t_elapsed:.2f}s")
 
    # ── Insertion only (from random) ──
    t0 = time.time()
    df_i = insertion_sort_by_majority(df_rand)
    t_elapsed = time.time() - t0
    ev = evaluate(df_i)
    results["Insert_only"] = {**ev, "time": t_elapsed}
    print(f"  [Insert only]     | F={ev['F']:.1f} | gap={ev['gap_pct']:.3f}% | {t_elapsed:.2f}s")
 
    # ── SA only (from random) ──
    t0 = time.time()
    if n > 10:
        df_s = simulated_annealing(df_rand, seed=0, verbose=False)
    else:
        df_s = df_rand.copy()
    t_elapsed = time.time() - t0
    ev = evaluate(df_s)
    results["SA_only"] = {**ev, "time": t_elapsed}
    print(f"  [SA only]         | F={ev['F']:.1f} | gap={ev['gap_pct']:.3f}% | {t_elapsed:.2f}s")
 
    # ── SW only (from random) ──
    t0 = time.time()
    df_w = refine_after_sa(df_rand, window_size=20, verbose=False)
    t_elapsed = time.time() - t0
    ev = evaluate(df_w)
    results["SW_only"] = {**ev, "time": t_elapsed}
    print(f"  [SW only]         | F={ev['F']:.1f} | gap={ev['gap_pct']:.3f}% | {t_elapsed:.2f}s")

    # ── SW only (based on borda) ──
    t0 = time.time()
    df_b_for_sw = borda_sort(df_sub)
    df_sw_borda = refine_after_sa(df_b_for_sw, window_size=20, verbose=False)
    t_elapsed = time.time() - t0
    ev = evaluate(df_sw_borda)
    results["SW_only_borda"] = {**ev, "time": t_elapsed}
    print(f"  [SW only (borda)] | F={ev['F']:.1f} | gap={ev['gap_pct']:.3f}% | {t_elapsed:.2f}s")
 
    return results
 
 
# ======================================================================
# 汇总打印
# ======================================================================
 
def print_summary_table(all_results: dict):
    """将所有 SCC 的实验结果汇总成表格打印。"""
 
    print(f"\n{'='*80}")
    print("ABLATION SUMMARY")
    print(f"{'='*80}")
 
    for scc_label, experiments in all_results.items():
        print(f"\n┌─ SCC: {scc_label} ─────────────────────────────────────────────────")
 
        for exp_name, configs in experiments.items():
            print(f"│")
            print(f"│  {exp_name}:")
            print(f"│  {'Config':<22s} {'F':>12s} {'Gap%':>10s} {'StepTime':>10s} {'TotalTime':>10s}")
            print(f"│  {'─'*64}")
 
            # 取 Full pipeline 的 F 作为基准（如果有）
            baseline_F = None
            for cfg_name, vals in configs.items():
                if cfg_name in ("Full", "4_Full"):
                    baseline_F = vals["F"]
                    break
            if baseline_F is None:
                # 取最后一个 config 的 F
                baseline_F = list(configs.values())[-1]["F"]
 
            for cfg_name, vals in configs.items():
                F = vals["F"]
                gap = vals["gap_pct"]
                st = vals.get("step_time", vals.get("time", 0))
                tt = vals.get("time", 0)
                delta = F - baseline_F
 
                delta_str = f"(+{delta:.0f})" if delta > 0 else ""
                print(f"│  {cfg_name:<22s} {F:>12.1f} {gap:>9.3f}% "
                      f"{st:>9.2f}s {tt:>9.2f}s  {delta_str}")
 
        print(f"└{'─'*72}")

if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "./output/university_rankings.csv"
    m = int(sys.argv[2]) if len(sys.argv) > 2 else 11
    window_size = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    assert 3 <= window_size <= 50, "Window size must be between 3 and 50 for meaningful optimisation."

    print(f"Ablation Study")
    print(f"Data: {csv_path}, m={m}")
    print(f"{'='*80}")
 
    # ── SCC 分解 ──
    t_scc_start = time.time()
    topo, groups, labels = scc_grouping(csv_path, m)
    t_scc = time.time() - t_scc_start
    print(f"\n[SCC] Decomposition time: {t_scc:.2f}s")
    print(f"[SCC] Components: {len(topo)}, "
          f"sizes: {sorted([len(groups[s]) for s in topo], reverse=True)}")
 
    df_rank_all = load_rank_data(csv_path, m)
 
    # ── 对每个非平凡 SCC 运行三组实验 ──
    all_results = {}
 
    for scc_id in topo:
        members = groups[scc_id]
        n = len(members)
        if n <= 1:
            continue
 
        scc_label = f"SCC_{scc_id}(n={n})"
        df_sub = extract_subset(df_rank_all, members)
 
        experiments = OrderedDict()
 
        # 实验 A：累积
        experiments["Cumulative"] = run_cumulative(df_sub, scc_label, m)
 
        # 实验 B：Skip-one（仅对 n>50 有意义，否则 SA 被跳过）
        if n > 20:
            experiments["Skip-One"] = run_skip_one(df_sub, scc_label, m)
 
        # 实验 C：单步独立能力（仅对较大的 SCC）
        if n > 20:
            experiments["Isolated"] = run_isolated(df_sub, scc_label, m)
 
        all_results[scc_label] = experiments
 
    # ── 汇总 ──
    print_summary_table(all_results)
 
    # ── 输出为 CSV 以便进一步分析 ──
    rows = []
    for scc_label, experiments in all_results.items():
        for exp_name, configs in experiments.items():
            for cfg_name, vals in configs.items():
                rows.append({
                    "scc": scc_label,
                    "experiment": exp_name,
                    "config": cfg_name,
                    "F": vals["F"],
                    "lower_bound": vals["lower_bound"],
                    "gap_pct": vals["gap_pct"],
                    "step_time": vals.get("step_time", None),
                    "total_time": vals.get("time", None),
                })
    df_out = pd.DataFrame(rows)
    out_path = "./output/ablation_results.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")