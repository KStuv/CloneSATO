import numpy as np
import torch
import networkx as nx
import community as community_louvain


def build_sna_community(
    edge_index,
    num_nodes,
    device,
    resolution=1.0,
    seed=42
):
    """
    Build SNA community features from training edges only.

    Returns:
        community_id: [num_nodes]
        density: [num_communities, num_communities]
        bridge_score: [num_nodes]
    """

    # 1. Build undirected graph
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))

    edges = edge_index.detach().cpu().numpy()

    graph.add_edges_from(
        (int(u), int(v))
        for u, v in edges.T
        if u != v
    )

    # 2. Louvain community detection. An edgeless graph has no meaningful
    # partition for Louvain, so place all isolated nodes in one community.
    if num_nodes == 0:
        raise ValueError("num_nodes must be greater than zero")
    if graph.number_of_edges() == 0:
        partition = {node: 0 for node in range(num_nodes)}
    else:
        partition = community_louvain.best_partition(
            graph,
            resolution=resolution,
            random_state=seed
        )

    # 3. Convert community labels to consecutive IDs
    unique_communities = sorted(set(partition.values()))

    community_map = {
        old_id: new_id
        for new_id, old_id in enumerate(unique_communities)
    }

    labels = np.array([
        community_map[partition[node]]
        for node in range(num_nodes)
    ], dtype=np.int64)

    num_communities = len(unique_communities)

    # 4. Calculate community sizes
    sizes = np.bincount(
        labels,
        minlength=num_communities
    ).astype(np.float64)

    # 5. Calculate inter-community edge counts
    edge_counts = np.zeros(
        (num_communities, num_communities),
        dtype=np.float64
    )

    for u, v in graph.edges():
        cu = labels[u]
        cv = labels[v]

        edge_counts[cu, cv] += 1

        if cu != cv:
            edge_counts[cv, cu] += 1

    # 6. Calculate community pair density
    density = np.zeros_like(edge_counts)

    for a in range(num_communities):
        for b in range(num_communities):

            if a == b:
                possible = sizes[a] * (sizes[a] - 1) / 2
                actual = edge_counts[a, a]

            else:
                possible = sizes[a] * sizes[b]
                actual = edge_counts[a, b]

            if possible > 0:
                density[a, b] = actual / possible

    # 7. Calculate bridge node score
    bridge_score = np.zeros(num_nodes, dtype=np.float32)

    for node in range(num_nodes):
        neighbors = list(graph.neighbors(node))

        if len(neighbors) == 0:
            continue

        external_neighbors = sum(
            labels[node] != labels[neighbor]
            for neighbor in neighbors
        )

        bridge_score[node] = (
            external_neighbors / len(neighbors)
        )

    return (
        torch.tensor(labels, dtype=torch.long, device=device),
        torch.tensor(density, dtype=torch.float32, device=device),
        torch.tensor(bridge_score, dtype=torch.float32, device=device)
    )


def get_triadic_features(i, j, u, community):
    """
    Extract SNA features for triad (i, j, u).

    i, j, u: 1-D LongTensor
    """

    labels, density, bridge_score = community

    ci = labels[i]
    cj = labels[j]
    cu = labels[u]

    affinity_ij = (ci == cj).float()
    affinity_iu = (ci == cu).float()
    affinity_ju = (cj == cu).float()

    density_ij = density[ci, cj]
    density_iu = density[ci, cu]
    density_ju = density[cj, cu]

    bridge_u = bridge_score[u]

    features = torch.stack([
        affinity_ij,
        affinity_iu,
        affinity_ju,
        density_ij,
        density_iu,
        density_ju,
        bridge_u
    ], dim=-1)

    return features