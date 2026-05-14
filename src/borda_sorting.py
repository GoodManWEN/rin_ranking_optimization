import pandas as pd

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