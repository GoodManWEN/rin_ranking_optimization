import pandas as pd

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