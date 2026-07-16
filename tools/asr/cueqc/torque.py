"""Torque Clustering (TORC) — single-file Python port.

Ported from the official MATLAB implementation by Jie Yang (UTS) and the
community Python adaptation (Cognet-74/TorqueClusteringPy) referenced in the
official README. TORC is a parameter-free clustering algorithm that decides the
number of clusters autonomously from the distance matrix alone; it only exposes
``K`` (0 = automatic) and ``detect_noise``.

Reference:
    Jie Yang and Chin-Teng Lin, "Autonomous clustering by fast find of mass and
    distance peaks," IEEE TPAMI, DOI: 10.1109/TPAMI.2025.3535743.

License: CC BY-NC-SA 4.0 (academic and research use only; commercial use
prohibited). Attribution to Jie Yang is retained here and in the project
docs/HISTORY.md. The upstream MATLAB/.p core is obfuscated; this module reproduces
the algorithmic behaviour documented in the paper and the readable helpers
(mindisttwinsloc / ps2psdist / Qac / inipd) shipped alongside it.

Dependencies: numpy and scipy.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


# ---------------------------------------------------------------------------
# Helpers (direct translations of the readable MATLAB sub-functions)
# ---------------------------------------------------------------------------

def _ps2psdist(loc1: np.ndarray, loc2: np.ndarray, dm: sp.spmatrix | np.ndarray) -> float:
    """Minimum pairwise distance between two point sets (MATLAB ps2psdist)."""
    loc1 = np.asarray(loc1, dtype=np.intp)
    loc2 = np.asarray(loc2, dtype=np.intp)
    if loc1.size == 0 or loc2.size == 0:
        return float("inf")
    sub = dm[np.ix_(loc1, loc2)]
    if sp.issparse(sub):
        sub = sub.toarray()
    return float(np.min(sub))


def _mindisttwinsloc(loc1: np.ndarray, loc2: np.ndarray, dm: sp.spmatrix | np.ndarray):
    """Indices (within loc1, loc2) of the minimum-distance cross pair."""
    loc1 = np.asarray(loc1, dtype=np.intp)
    loc2 = np.asarray(loc2, dtype=np.intp)
    sub = dm[np.ix_(loc1, loc2)]
    if sp.issparse(sub):
        sub = sub.toarray()
    flat = int(np.argmin(sub))
    a, b = np.unravel_index(flat, sub.shape)
    return int(loc1[a]), int(loc2[b])


def _qac(sort_p: np.ndarray) -> np.ndarray:
    """Successive ratio vector used by Nab_dec (MATLAB Qac)."""
    n = len(sort_p)
    ind = np.zeros(n, dtype=np.float64)
    if n > 1:
        denom = sort_p[1:].copy()
        safe = np.where(denom == 0.0, np.nan, denom)
        ind[: n - 1] = sort_p[: n - 1] / safe
    ind[n - 1] = np.nan
    return ind


def _unique_z(z: np.ndarray, ljmat: sp.spmatrix):
    """Deduplicate recorded links and drop the redundant ones from the graph."""
    if z.size == 0:
        return z, ljmat
    sortrow = np.sort(z[:, [0, 1]], axis=1)
    _, order = np.unique(sortrow, axis=0, return_index=True)
    order = np.sort(order)
    newz = z[order].copy()
    sortrow_y = np.sort(z[:, [2, 3]], axis=1)
    keep_mask = np.zeros(z.shape[0], dtype=bool)
    keep_mask[order] = True
    redundant = sortrow_y[~keep_mask]
    if redundant.size > 0:
        ljmat = ljmat.tolil()
        for r in redundant:
            i1, i2 = int(r[0]), int(r[1])
            ljmat[i1, i2] = 0
            ljmat[i2, i1] = 0
        ljmat = ljmat.tocsr()
    return newz, ljmat


def _update_ljmat(old_ljmat, neiborloc, community, commu_dm, g, all_dm):
    """Record the cross-link + mass/distance for every community neighbour edge."""
    community_num = len(community)
    if community_num == 0:
        return old_ljmat, np.empty((0, 7))
    pd = len(community[0])
    if pd > 1:
        rows = [None] * community_num
        th = 0
        for i in range(community_num):
            nb = neiborloc[i]
            if nb is None:
                continue
            linkloc1, linkloc2 = _mindisttwinsloc(community[i], community[nb], all_dm)
            old_ljmat[linkloc1, linkloc2] = 1
            old_ljmat[linkloc2, linkloc1] = 1
            rows[th] = (
                min(community[i]), min(community[nb]),
                linkloc1, linkloc2,
                len(community[i]), len(community[nb]),
                float(commu_dm[i, nb]),
            )
            th += 1
        cutlinkpower = np.array([r for r in rows[:th] if r is not None], dtype=np.float64).reshape(-1, 7)
    else:
        cutlinkpower = np.zeros((community_num, 7), dtype=np.float64)
        for i in range(community_num):
            nb = neiborloc[i]
            linkloc1 = community[i][0]
            linkloc2 = community[nb][0]
            cutlinkpower[i] = (
                linkloc1, linkloc2, linkloc1, linkloc2,
                1, 1, float(commu_dm[i, nb]),
            )
        old_ljmat = g.tocsr() if sp.issparse(g) else g
    return old_ljmat, cutlinkpower


def _nab_dec(p: np.ndarray, mass: np.ndarray, r: np.ndarray, florderloc: np.ndarray,
             *, use_std_adjustment: bool = True, adjustment_factor: float = 0.5) -> int:
    """Decide how many abnormal merges to cut (MATLAB Nab_dec).

    TORC treats the *high-torque* merges as abnormal (a heavy mass pulled across
    a large distance = two distinct clusters wrongly stitched together) and cuts
    the leading run of them to recover natural clusters.

    Torque spans many orders of magnitude, so the cut point is located via the
    successive ratio gaps ``p[i]/p[i+1]`` of the descending torque array. The
    bulk of within-cluster merges sits at ratios near 1; abnormal boundaries show
    a sharp ratio peak. The cut count is the position of the *last* ratio that
    clears a significance threshold above the bulk median, so a long run of
    genuine outliers is fully removed rather than only the single largest gap.
    ``adjustment_factor`` scales the threshold. ``florderloc`` (initial-layer NN
    links) is excluded since those are genuine merges, not outliers.
    """
    if p.size < 2:
        return 1
    order_desc = np.argsort(-p, kind="mergesort")
    sort_p = p[order_desc]
    fl = set(int(x) for x in np.asarray(florderloc, dtype=np.int64).ravel())
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = sort_p[:-1] / sort_p[1:]
    finite = ratios[np.isfinite(ratios) & (ratios > 0)]
    if finite.size == 0:
        return 1
    bulk = float(np.median(finite))
    # Significance threshold above the bulk. adjustment_factor scales how sharp a
    # boundary must be to count; 0.5 -> ratio >= bulk + 0.5*(max-bulk) heuristic.
    span = float(np.max(finite) - bulk)
    threshold = bulk + adjustment_factor * span
    # Walk descending and keep the last index whose ratio clears the threshold —
    # that marks the bottom of the abnormal pile.
    cut = 0
    for i in range(ratios.size):
        if i in fl:
            continue
        ratio = ratios[i]
        if np.isfinite(ratio) and ratio >= threshold:
            cut = i + 1
    return max(1, cut)


def _community_min_distance_matrix(
    dm: np.ndarray,
    point_to_comm: np.ndarray,
    comm_num: int,
    *,
    chunk_rows: int = 256,
) -> np.ndarray:
    """Minimum point-to-point distance between current TORC communities.

    This is the vectorized equivalent of the original nested
    ``_ps2psdist(community[i], community[j])`` loop. It preserves the same
    min-distance community metric while avoiding Python-level O(c^2) loops.
    """
    inter = np.full((comm_num, comm_num), np.inf, dtype=np.float64)
    labels_right = point_to_comm[None, :]
    n = dm.shape[0]
    for start in range(0, n, chunk_rows):
        end = min(n, start + chunk_rows)
        labels_left = point_to_comm[start:end, None]
        np.minimum.at(inter, (labels_left, labels_right), dm[start:end, :])
    np.fill_diagonal(inter, np.inf)
    return inter


def _merge_layers_fast(
    dm: np.ndarray,
    *,
    max_layer: int,
) -> list[np.ndarray]:
    """Return TORC merge-hierarchy partitions without building the cut tree.

    Merge-layer mode never uses torque-gap pruning. The requested partitions are
    fully determined by repeated 1NN / mass-rule community merges, so this fast
    path returns the layer sequence directly and stops at ``max_layer``.
    """
    if max_layer < 0:
        raise ValueError("max_layer must be non-negative")
    n = dm.shape[0]
    if n <= 1:
        return [np.zeros(n, dtype=np.int64)]

    diag = np.diag(dm).copy()
    np.fill_diagonal(dm, np.inf)
    try:
        nearest = np.argmin(dm, axis=1).astype(np.int64)
        graph = sp.csr_matrix(
            (
                np.ones(n * 2, dtype=np.float64),
                (
                    np.concatenate([np.arange(n, dtype=np.int64), nearest]),
                    np.concatenate([nearest, np.arange(n, dtype=np.int64)]),
                ),
            ),
            shape=(n, n),
        )
        _count, point_labels = connected_components(graph, directed=False)
        point_labels = point_labels.astype(np.int64, copy=False)
        layer_labels = [point_labels.copy()]
        if max_layer == 0:
            return layer_labels

        while len(layer_labels) <= max_layer:
            _unique, point_to_comm = np.unique(point_labels, return_inverse=True)
            point_to_comm = point_to_comm.astype(np.int64, copy=False)
            comm_num = int(point_to_comm.max()) + 1
            sizes = np.bincount(point_to_comm, minlength=comm_num)
            if comm_num <= 1:
                break

            inter = _community_min_distance_matrix(dm, point_to_comm, comm_num)
            # Mass rule: community i may attach only to a community j that is
            # not smaller. This matches the reference loop's first nearest
            # candidate with sizes[i] <= sizes[j].
            inter[sizes[None, :] < sizes[:, None]] = np.inf
            nearest_comm = np.argmin(inter, axis=1).astype(np.int64)
            rows = np.arange(comm_num, dtype=np.int64)
            valid = np.isfinite(inter[rows, nearest_comm])
            if not np.any(valid):
                break

            graph = sp.csr_matrix(
                (
                    np.ones(int(np.sum(valid)) * 2, dtype=np.float64),
                    (
                        np.concatenate([rows[valid], nearest_comm[valid]]),
                        np.concatenate([nearest_comm[valid], rows[valid]]),
                    ),
                ),
                shape=(comm_num, comm_num),
            )
            _count, comm_bins = connected_components(graph, directed=False)
            next_labels = comm_bins[point_to_comm].astype(np.int64, copy=False)
            if np.array_equal(next_labels, point_labels):
                break
            point_labels = next_labels
            layer_labels.append(point_labels.copy())

        return layer_labels
    finally:
        np.fill_diagonal(dm, diag)


def _merge_layer_fast(
    dm: np.ndarray,
    *,
    merge_layer: int,
) -> tuple[np.ndarray, list[int]]:
    """Return one TORC merge-hierarchy partition without building the cut tree."""
    layer_labels = _merge_layers_fast(dm, max_layer=merge_layer)
    if merge_layer >= len(layer_labels):
        raise ValueError(
            f"merge_layer {merge_layer} out of range; "
            f"available layers 0..{len(layer_labels) - 1}"
        )
    return _relabel_by_size(layer_labels[merge_layer]), [
        int(len(np.unique(labels))) for labels in layer_labels
    ]


def torque_merge_layer_preview(
    all_dm: np.ndarray | sp.spmatrix,
    *,
    max_layer: int = 8,
) -> list[dict[str, Any]]:
    """Summarize TORC merge layers for choosing a human-auditable cluster level."""
    if max_layer < 0:
        raise ValueError("max_layer must be non-negative")
    if isinstance(all_dm, sp.spmatrix):
        dm = all_dm.toarray().astype(np.float64, copy=False)
    else:
        dm = np.asarray(all_dm, dtype=np.float64)
    if dm.ndim != 2 or dm.shape[0] != dm.shape[1]:
        raise ValueError("distance matrix must be square")
    preview: list[dict[str, Any]] = []
    for layer, labels in enumerate(_merge_layers_fast(dm.copy(), max_layer=max_layer)):
        _unique, counts = np.unique(labels, return_counts=True)
        preview.append(
            {
                "layer": layer,
                "cluster_count": int(len(counts)),
                "cluster_size_min": int(counts.min()) if counts.size else 0,
                "cluster_size_max": int(counts.max()) if counts.size else 0,
                "cluster_size_avg": round(float(counts.mean()) if counts.size else 0.0, 4),
            }
        )
    return preview


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def torque_clustering(
    all_dm: np.ndarray | sp.spmatrix,
    *,
    k: int = 0,
    detect_noise: bool = False,
    use_std_adjustment: bool = True,
    adjustment_factor: float | None = None,
    merge_layer: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Cluster from a square distance matrix using Torque Clustering.

    Parameters mirror the official signature ``TorqueClustering(ALL_DM, K, isnoise)``:
    ``k=0`` selects automatic cluster-count determination; ``k>0`` forces that many
    clusters. ``detect_noise`` flags outliers (label 0 in the noise-aware output).
    ``adjustment_factor`` is used only by the legacy torque-gap cut. ``None``
    uses the fixed default ``0.5``. Merge-layer mode ignores it.

    ``merge_layer`` selects a partition from the TORC merge hierarchy instead of
    the torque-gap cut on the final tree. Layer 0 is the initial 1-NN layer;
    layer 1 is after one merge pass; etc. This is far more stable than the
    factor-sensitive final-tree cut on heavy-tailed distance distributions, and
    each layer is a natural coarsening. When set, the cut logic is skipped.

    Returns ``(labels, labels_with_noise, diagnostics)`` where ``labels`` is the
    noise-free assignment (0-based contiguous) and ``diagnostics`` carries the
    number of clusters, the number of cut links, the mass/R/torque arrays, and
    (when merge_layer is used) the per-layer cluster counts.
    """
    if all_dm is None:
        raise ValueError("distance matrix is required")
    dense_dm: np.ndarray | None = None
    if isinstance(all_dm, sp.spmatrix):
        dm_sparse = all_dm.tocsr().astype(np.float64)
        n = dm_sparse.shape[0]
    else:
        arr = np.asarray(all_dm, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError("distance matrix must be square")
        n = arr.shape[0]
        dense_dm = arr
        dm_sparse = sp.csr_matrix(arr)
    if n <= 1:
        if adjustment_factor is None:
            adjustment_factor = 0.5
        return np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64), {
            "cluster_count": max(1, n), "cut_count": 0,
            "noise_count": 0,
            "mass": np.array([]), "r": np.array([]), "torque": np.array([]),
            "k_requested": k, "detect_noise": detect_noise,
            "adjustment_factor": float(adjustment_factor),
        }

    if merge_layer is not None:
        if dense_dm is None:
            dense_dm = dm_sparse.toarray()
        labels, layer_counts = _merge_layer_fast(dense_dm, merge_layer=merge_layer)
        diag = {
            "cluster_count": int(len(np.unique(labels))),
            "cut_count": 0,
            "noise_count": 0,
            "mass": np.array([]), "r": np.array([]), "torque": np.array([]),
            "k_requested": k, "detect_noise": detect_noise,
            "merge_layer": merge_layer,
            "layer_cluster_counts": layer_counts,
            "fast_merge_layer": True,
        }
        return labels, labels.copy(), diag

    if adjustment_factor is None:
        adjustment_factor = 0.5

    with np.errstate(all="ignore"):
        link_adj = sp.lil_matrix((n, n), dtype=np.float64)
        community: list[list[int]] = [[int(i)] for i in range(n)]
        commu_dm = dm_sparse
        cutlinkpower_all: list[np.ndarray] = []
        # Per-layer original-point partitions (layer 0 = initial 1-NN layer).
        # Used when merge_layer is set: each layer is a natural coarsening and far
        # more stable than the factor-sensitive torque-gap cut on the final tree.
        layer_labels: list[np.ndarray] = []

        # --- initial layer: connect each point to its 1-nearest neighbour ---
        graph = sp.lil_matrix((n, n), dtype=np.float64)
        neiborloc: list[int] = [0] * n
        dm_dense_rows = dense_dm if dense_dm is not None else (dm_sparse.toarray() if n <= 2048 else None)
        for i in range(n):
            row = dm_dense_rows[i] if dm_dense_rows is not None else dm_sparse[i].toarray().ravel()
            row[i] = np.inf
            min_idx = int(np.argmin(row))
            graph[i, min_idx] = 1
            graph[min_idx, i] = 1
            neiborloc[i] = min_idx
        graph = graph.tocsr()
        _, bins = connected_components(graph, directed=False)
        layer_labels.append(bins.astype(np.int64).copy())
        link_adj, cutlinkpower = _update_ljmat(link_adj, neiborloc, community, commu_dm, graph, dm_sparse)
        cutlinkpower, link_adj = _unique_z(cutlinkpower, link_adj)
        firstlayer_num = cutlinkpower.shape[0] if cutlinkpower.size else 0
        if cutlinkpower.size:
            cutlinkpower_all.append(cutlinkpower)

        # --- iterative merging: connect each community to its nearest >=-mass neighbour ---
        previous_unique = 0
        while True:
            idx = bins.copy()
            uni = np.unique(idx)
            new_community: list[list[int]] = []
            for label in uni:
                members = np.where(idx == label)[0]
                merged: list[int] = []
                for m in members:
                    merged.extend(community[m])
                new_community.append(merged)
            community = new_community
            comm_num = len(community)

            point_to_comm = np.empty(n, dtype=np.int64)
            for ci, members in enumerate(community):
                point_to_comm[np.asarray(members, dtype=np.int64)] = ci
            if dense_dm is not None:
                inter_dense = _community_min_distance_matrix(dense_dm, point_to_comm, comm_num)
            else:
                inter_dense = _community_min_distance_matrix(dm_sparse.toarray(), point_to_comm, comm_num)
            inter_dm = sp.csr_matrix(inter_dense)

            graph = sp.lil_matrix((comm_num, comm_num), dtype=np.float64)
            neiborloc2: list = [None] * comm_num
            sizes = np.array([len(c) for c in community])
            for i in range(comm_num):
                row = inter_dense[i]
                row[i] = np.inf
                order = np.argsort(row, kind="mergesort")
                for j in order:
                    if j == i:
                        continue
                    # only connect to a community that is not smaller (mass rule)
                    if sizes[i] <= sizes[j]:
                        graph[i, j] = 1
                        graph[j, i] = 1
                        neiborloc2[i] = int(j)
                        break
            graph = graph.tocsr()
            _, bins = connected_components(graph, directed=False)
            # Record this layer's original-point partition by expanding
            # community -> component label back to every original point.
            point_label = np.empty(n, dtype=np.int64)
            for ci, members in enumerate(community):
                pt = int(bins[ci])
                for member in members:
                    point_label[member] = pt
            layer_labels.append(point_label.copy())
            link_adj, cutlinkpower = _update_ljmat(link_adj, neiborloc2, community, inter_dm, graph, dm_sparse)
            cutlinkpower, link_adj = _unique_z(cutlinkpower, link_adj)
            if cutlinkpower.size:
                cutlinkpower_all.append(cutlinkpower)
            unique_bins = np.unique(bins)
            if len(unique_bins) == 1 or len(unique_bins) == previous_unique:
                break
            previous_unique = len(unique_bins)

        if not cutlinkpower_all:
            empty_mass = np.array([])
            return np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64), {
                "cluster_count": 1, "cut_count": 0,
                "noise_count": 0,
                "mass": empty_mass, "r": empty_mass, "torque": empty_mass,
                "k_requested": k, "detect_noise": detect_noise,
                "adjustment_factor": float(adjustment_factor),
            }

        cutlink_all = np.vstack([c for c in cutlinkpower_all if c.size]).astype(np.float64)
        mass = cutlink_all[:, 4] * cutlink_all[:, 5]
        r = cutlink_all[:, 6] ** 2
        torque = mass * r

        order_desc = np.argsort(-torque, kind="mergesort")
        order_rank = np.argsort(order_desc, kind="mergesort")
        # firstlayer_loc = positions (in descending-torque order) of the initial
        # nearest-neighbour links. Those merges formed the genuine base clusters
        # and must be excluded from the abnormal-cut decision.
        firstlayer_mask = np.zeros(cutlink_all.shape[0], dtype=bool)
        firstlayer_mask[:firstlayer_num] = True
        firstlayer_loc = order_rank[firstlayer_mask] if firstlayer_num > 0 else np.array([], dtype=np.int64)

        # --- decide cut count (torque gap or user k) ---
        if k == 0:
            cut_count = _nab_dec(torque, mass, r, firstlayer_loc,
                                 use_std_adjustment=use_std_adjustment,
                                 adjustment_factor=adjustment_factor)
        else:
            cut_count = max(1, k - 1)
        cut_count = int(min(cut_count, len(order_desc)))

        # --- cut links and derive noise-free labels ---
        # Build a clean merge tree directly from the recorded point-pair links
        # (cutlink_all columns 2,3 = linkloc1, linkloc2 = original point indices).
        # Each merge recorded one representative cross edge; cutting the
        # high-torque edges splits the tree into the natural clusters.
        rows_idx = cutlink_all[:, 2].astype(np.int64)
        cols_idx = cutlink_all[:, 3].astype(np.int64)
        tree = sp.csr_matrix(
            (np.ones(rows_idx.size, dtype=np.float64), (rows_idx, cols_idx)),
            shape=(n, n),
        )
        tree = tree + tree.T  # symmetrise
        indices_to_cut = [int(order_desc[i]) for i in range(cut_count)]
        if indices_to_cut:
            cut_mask = np.zeros(rows_idx.size, dtype=bool)
            cut_mask[indices_to_cut] = True
            tree = tree.tolil()
            for gi in indices_to_cut:
                r0 = int(rows_idx[gi])
                c0 = int(cols_idx[gi])
                tree[r0, c0] = 0
                tree[c0, r0] = 0
            tree = tree.tocsr()
        _, labels = connected_components(tree, directed=False)

        labels_with_noise = labels.copy()
        if detect_noise:
            r_mean = float(np.mean(r))
            mass_mean = float(np.mean(mass))
            r_mass = r / np.where(mass == 0, 1e-12, mass)
            r_mass_mean = float(np.mean(r_mass))
            eps = 1e-10
            noise_loc = np.intersect1d(
                np.intersect1d(np.where(r >= r_mean - eps)[0], np.where(mass <= mass_mean + eps)[0]),
                np.where(r_mass >= r_mass_mean - eps)[0],
            )
            cut_set = set(indices_to_cut).union(set(int(x) for x in noise_loc))
            tree2 = tree.tolil()
            for gi in cut_set:
                r0 = int(rows_idx[gi])
                c0 = int(cols_idx[gi])
                tree2[r0, c0] = 0
                tree2[c0, r0] = 0
            tree2 = tree2.tocsr()
            _, labels2 = connected_components(tree2, directed=False)
            labels_with_noise = _final_label(labels, labels2)

        # relabel contiguously (0-based) by size-descending for stability
        labels = _relabel_by_size(labels)
        labels_with_noise = _relabel_by_size(labels_with_noise, noise_zero=True)

    diag = {
        "cluster_count": int(len(np.unique(labels))),
        "cut_count": cut_count,
        "mass": mass,
        "r": r,
        "torque": torque,
        "k_requested": k,
        "detect_noise": detect_noise,
        "noise_count": int(np.sum(labels_with_noise < 0)),
        "adjustment_factor": float(adjustment_factor),
    }
    return labels, labels_with_noise, diag


# ---------------------------------------------------------------------------
# Small local helpers
# ---------------------------------------------------------------------------

def _final_label(labels1: np.ndarray, labels2: np.ndarray) -> np.ndarray:
    """Keep each cluster's largest consistent sub-group; rest become noise (-1)."""
    out = labels1.astype(np.int64).copy()
    for lab in np.unique(labels1):
        class_loc = np.where(labels1 == lab)[0]
        best = np.array([], dtype=np.int64)
        for lab2 in np.unique(labels2):
            zj_loc = np.where(labels2 == lab2)[0]
            if zj_loc.size and np.all(np.isin(zj_loc, class_loc)) and zj_loc.size > best.size:
                best = zj_loc
        noise = np.setdiff1d(class_loc, best)
        out[noise] = -1
    return out


def _relabel_by_size(labels: np.ndarray, *, noise_zero: bool = False) -> np.ndarray:
    """Renumber clusters 0..K-1 by descending size; noise (label<0) → -1."""
    out = np.full_like(labels, -1)
    mask = labels >= 0
    if not mask.any():
        return out
    uniq, counts = np.unique(labels[mask], return_counts=True)
    order = np.argsort(-counts, kind="mergesort")
    remap = {int(uniq[u]): i for i, u in enumerate(order)}
    for i in range(len(labels)):
        if mask[i]:
            out[i] = remap[int(labels[i])]
    return out
