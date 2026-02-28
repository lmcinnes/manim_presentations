# type: ignore
import numpy as np
import hdbscan
from evoc.clustering import (
    build_kdtree,
    parallel_boruvka,
    condense_tree,
    compute_total_persistence,
    min_cluster_size_barcode,
    mst_to_linkage_tree,
    extract_leaves,
    get_cluster_label_vector,
    get_point_membership_strength_vector,
    find_peaks,
    mask_condensed_tree,
)
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "extradata"

_LAMBDA_SCALE = 100.0


def lambda_to_density(lam, lambda_scale=_LAMBDA_SCALE):
    return np.exp(-lambda_scale / lam)


def pdf_order(cluster_tree, current_node):
    children = cluster_tree[cluster_tree["parent"] == current_node]["child"]
    if len(children) == 0:
        return [current_node]
    sorted_children = np.concatenate([children[0::2], children[1::2][::-1]])
    return sum([pdf_order(cluster_tree, child) for child in sorted_children], [])


def descendant_points(ctree, cluster_num):
    result = ctree["child"][
        (ctree["parent"] == cluster_num) & (ctree["child_size"] == 1)
    ].tolist()
    child_clusters = ctree["child"][
        (ctree["parent"] == cluster_num) & (ctree["child_size"] > 1)
    ]
    for c in child_clusters:
        result.extend(descendant_points(ctree, c))
    return result


def find_permutation(arr1, arr2):
    sorter = np.argsort(arr1, kind="stable")
    return sorter[np.searchsorted(arr1, arr2, sorter=sorter)]


def fit_hdbscan(data, **kwargs):
    """Fit HDBSCAN and return (model, ctree, points_in_pdf_order).

    Default parameters match the precomputation settings; callers can
    override any HDBSCAN keyword argument.
    """
    defaults = dict(
        min_samples=5,
        min_cluster_size=15,
        cluster_selection_method="leaf",
    )
    defaults.update(kwargs)
    model = hdbscan.HDBSCAN(**defaults).fit(data)
    ctree = model.condensed_tree_.to_numpy()
    points_in_pdf_order = pdf_order(ctree, ctree["parent"].min())
    return model, ctree, points_in_pdf_order


def density_profile_for_cluster(ctree, cluster_num, order, lambda_scale=_LAMBDA_SCALE):
    subtree = ctree[ctree["parent"] == cluster_num]
    singleton_children = subtree[subtree["child_size"] == 1]
    cluster_profile = np.vstack(
        [
            singleton_children["child"],
            lambda_to_density(singleton_children["lambda_val"], lambda_scale),
        ]
    ).T
    cluster_children = subtree[subtree["child_size"] > 1]
    extra_cluster_profiles = []
    for row in cluster_children:
        points = descendant_points(ctree, row["child"])
        extra_cluster_profiles.append(
            np.vstack(
                [
                    points,
                    np.full(
                        len(points), lambda_to_density(row["lambda_val"], lambda_scale)
                    ),
                ]
            ).T
        )
    cluster_profile = np.vstack([cluster_profile] + extra_cluster_profiles)
    size = cluster_profile.shape[0]
    missing_indices = np.setdiff1d(
        np.arange(order.shape[0]), cluster_profile.T[0].astype(np.int32)
    )
    parent_row = ctree[ctree["child"] == cluster_num]
    min_val = (
        lambda_to_density(parent_row["lambda_val"], lambda_scale)
        if len(parent_row) == 1
        else 0.0
    )
    cluster_profile = np.vstack(
        [
            cluster_profile,
            np.vstack([missing_indices, np.full(missing_indices.shape[0], min_val)]).T,
        ]
    )
    cluster_profile_final = cluster_profile[
        find_permutation(cluster_profile.T[0].astype(np.int32), order)
    ]
    return cluster_profile_final.T[1], size


def compute_density_profiles(base_data):
    _, ctree, points_in_pdf_order = fit_hdbscan(base_data)
    order = np.array(points_in_pdf_order)
    cluster_tree = ctree[ctree["child_size"] > 1]
    clusters = np.unique(np.hstack([cluster_tree["parent"], cluster_tree["child"]]))

    profiles, sizes = [], []
    for cluster_id in clusters:
        profile, size = density_profile_for_cluster(ctree, cluster_id, order)
        profiles.append(profile)
        sizes.append(size)

    np.save(DATA_DIR / "cluster_density_profiles.npy", np.vstack(profiles))
    np.save(DATA_DIR / "cluster_sizes.npy", np.asarray(sizes))

    binary_tree = np.zeros((len(clusters), 2), dtype=np.int32)
    n_samples = base_data.shape[0]
    for row in cluster_tree:
        parent = row["parent"] - n_samples
        child = row["child"] - n_samples
        if binary_tree[parent, 0] > 0:
            binary_tree[parent, 1] = child
        else:
            binary_tree[parent, 0] = child

    np.save(DATA_DIR / "cluster_binary_tree.npy", binary_tree)


def compute_barcode(base_data):
    numba_tree = build_kdtree(base_data.astype(np.float32))
    edges = parallel_boruvka(numba_tree, 64, min_samples=5, reproducible=True)
    sorted_mst = edges[np.argsort(edges.T[2])]
    uncondensed_tree = mst_to_linkage_tree(sorted_mst)
    cond_tree = condense_tree(uncondensed_tree, 5)
    leaves = extract_leaves(cond_tree)
    clusters = get_cluster_label_vector(cond_tree, leaves, 0.0, base_data.shape[0])
    strengths = get_point_membership_strength_vector(cond_tree, leaves, clusters)
    mask = cond_tree.child >= base_data.shape[0]
    cluster_tree = mask_condensed_tree(cond_tree, mask)
    births, deaths, parents, lambda_deaths = min_cluster_size_barcode(
        cluster_tree, base_data.shape[0], 15
    )
    sizes, total_persistence = compute_total_persistence(births, deaths, lambda_deaths)
    peaks = find_peaks(total_persistence)

    valid_bars = deaths - births > 0
    valid_births = births[valid_bars]
    valid_deaths = deaths[valid_bars]
    valid_weights = lambda_deaths[valid_bars]
    valid_weights = np.maximum(valid_weights - valid_weights[1], 0) / np.max(
        valid_weights - valid_weights[1]
    )
    valid_lengths = valid_deaths - valid_births
    sort_order = np.argsort(valid_births)
    valid_births = valid_births[sort_order]
    valid_deaths = valid_deaths[sort_order]
    valid_lengths = valid_lengths[sort_order]
    valid_weights = valid_weights[sort_order]

    np.save(
        DATA_DIR / "barcode_bars.npy",
        np.vstack([valid_births, valid_deaths, valid_weights]).T,
    )
    np.save(
        DATA_DIR / "persistence_scores_trace.npy",
        np.vstack(
            [
                np.column_stack((sizes[:-1], sizes[1:])).reshape(-1),
                np.repeat(total_persistence[:-1], 2),
            ]
        ).T,
    )


def main():
    base_data = np.load(DATA_DIR / "base_data.npy")
    compute_density_profiles(base_data)
    compute_barcode(base_data)


if __name__ == "__main__":
    main()
