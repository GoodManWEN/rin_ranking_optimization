import sys
from scc_grouping import scc_grouping
from utils import load_rank_data, extract_subset, build_inversion_matrix_from_ranks
from insertion_sorting import insertion_sort_by_majority
from borda_sorting import borda_sort
from simulated_annealings import simulated_annealing_multi_run
from sliding_window_rin import refine_after_sa, _exhaustive_search 


def run_pipeline(csv_path, m, window_size):
    _, groups, _ = scc_grouping(csv_path, m)

    df_rank_all = load_rank_data(csv_path, m)
    for scc_id in sorted(groups.keys()):
        df_sub = extract_subset(df_rank_all, groups[scc_id])
        if len(groups[scc_id]) <= 1: # 此处待修改
            continue 
        elif len(groups[scc_id]) <= window_size:
            # 小型 SCC：直接暴力搜索
            # 进行Borda排序以提升暴力搜索效率
            df_insertion_sorted = borda_sort(df_sub)
            L_sub, unis_sub = build_inversion_matrix_from_ranks(df_insertion_sorted.values.astype(float)), list(df_insertion_sorted.index)
            _, best_perm = _exhaustive_search(L_sub, len(unis_sub))
            best_unis = [unis_sub[i] for i in best_perm]
            df_final = df_insertion_sorted.loc[best_unis]
        else:
            # broda排序 + 插入排序 + SA + 滑动窗口
            df_broda_sorted = borda_sort(df_sub)
            df_insertion_sorted = insertion_sort_by_majority(df_broda_sorted)
            if len(df_insertion_sorted) > 50:
                # 大型 SCC：SA + 滑动窗口
                df_annealing_sorted = simulated_annealing_multi_run(df_insertion_sorted, verbose=True, n_runs=3)
            else:
                df_annealing_sorted = df_insertion_sorted

            df_final = refine_after_sa(df_annealing_sorted, window_size=window_size, verbose=True)

        print(df_final)

        print(df_annealing_sorted)


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "./output/university_rankings.csv"
    m = int(sys.argv[2]) if len(sys.argv) > 2 else 11
    window_size = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    assert 3 <= window_size <= 50, "Window size must be between 3 and 50 for meaningful optimisation."

    run_pipeline(csv_path, m, window_size)
