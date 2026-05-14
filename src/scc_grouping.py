import pandas as pd
import numpy as np
from collections import defaultdict, deque
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

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