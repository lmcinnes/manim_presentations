import numpy as np
import time
import evoc

# import pandas as pd
import sklearn.cluster
import sklearn.datasets
import sklearn.metrics
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from umap.utils import tau_rand, tau_rand_int
from tqdm import tqdm
import numba
from sklearn.utils.extmath import randomized_svd


@numba.njit()
def procrustes_align(e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    e1_shift = e1 - np.sum(e1, axis=0) / e1.shape[0]
    e2_shift = e2 - np.sum(e2, axis=0) / e2.shape[0]
    e1_scale_factor = np.sqrt(np.mean(e1_shift**2))
    e2_scale_factor = np.sqrt(np.mean(e2_shift**2))
    e1_scaled = e1_shift / e1_scale_factor
    e2_scaled = e2_shift / e2_scale_factor
    covariance = e2_scaled.T @ e1_scaled
    u, s, vh = np.linalg.svd(covariance)
    if np.linalg.det(u @ vh) < 0:
        u[:, -1] *= -1
    rotation = u @ vh
    return rotation


INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1


@numba.njit(cache=True)
def umap_label_outliers(indptr, indices, labels, rng_state):
    n_rows = indptr.shape[0] - 1
    max_label = labels.max()
    num_labels = max(max_label + 1, 1)

    for i in range(n_rows):
        # Create a local rng state for this iteration
        local_rng_state = rng_state + i
        if labels[i] < 0:

            node_queue = [i]
            unlabelled = True
            n_iter = 0

            while (
                unlabelled and n_iter < 100 and len(node_queue) > 0
            ):  # Changed from 100

                n_iter += 1
                current_node = node_queue.pop()
                for k in range(indptr[current_node], indptr[current_node + 1]):
                    j = indices[k]
                    if labels[j] >= 0:
                        labels[i] = labels[j]
                        unlabelled = False
                        break
                    else:
                        node_queue.append(j)

            if unlabelled:
                labels[i] = tau_rand_int(local_rng_state) % num_labels

    return labels


@numba.njit(fastmath=True, parallel=True, cache=True)
def umap_label_prop_iteration(
    indptr,
    indices,
    data,
    labels,
    rng_state,
):
    n_rows = indptr.shape[0] - 1
    result = labels.copy()

    for i in numba.prange(n_rows):
        current_l = labels[i]
        if current_l >= 0:
            continue
        # Create a local rng state for this iteration
        local_rng_state = rng_state + i
        votes = {}
        for k in range(indptr[i], indptr[i + 1]):
            j = indices[k]
            l = labels[j]
            if l in votes:
                votes[l] += data[k]
            else:
                votes[l] = data[k]

        max_vote = 1
        tie_count = 1
        for l in votes:
            if l == -1:
                continue
            elif votes[l] > max_vote:
                max_vote = votes[l]
                result[i] = l
                tie_count = 1
            elif votes[l] == max_vote:
                tie_count += 1
                if current_l == -1:
                    result[i] = l
                elif tau_rand(local_rng_state) < 1.0 / tie_count:
                    result[i] = l
            else:
                continue

    return result


@numba.njit(cache=True)
def evoc_remap_labels(labels):
    mapping = {}
    unique_labels = np.unique(labels)
    if unique_labels[0] == -1:
        unique_labels = unique_labels[1:]
    for i, l in enumerate(unique_labels):
        mapping[l] = i
    next_label = i + 1
    for i in range(labels.shape[0]):
        if labels[i] < 0:
            labels[i] = next_label
            next_label += 1
        else:
            labels[i] = mapping[labels[i]]

    return labels


@numba.njit(cache=True)
def umap_remap_labels(labels):
    mapping = {}
    unique_labels = np.unique(labels)
    if unique_labels[0] == -1:
        unique_labels = unique_labels[1:]
    for i, l in enumerate(unique_labels):
        mapping[l] = i
    next_label = i + 1
    for i in range(labels.shape[0]):
        if labels[i] < 0:
            labels[i] = next_label
            next_label += 1
        else:
            labels[i] = mapping[labels[i]]

    return labels


@numba.njit(cache=True)
def umap_initialize_labels(labels, n_parts, rng_state):
    for i in range(n_parts):
        labels[tau_rand_int(rng_state) % labels.shape[0]] = i
    return labels


@numba.njit(cache=True)
def umap_initialize_labels_from_hubs(labels, n_parts, degrees):
    hubs = np.argsort(degrees)[-n_parts:]
    for i in range(n_parts):
        labels[hubs[i]] = i
    return labels


def make_epochs_per_sample(weights, n_epochs):
    result = np.full(weights.shape[0], n_epochs, dtype=np.float32)
    n_samples = np.maximum(n_epochs * (weights / weights.max()), 1.0)
    result = float(n_epochs) / np.float32(n_samples)
    return result


@numba.njit(
    "f4(f4[::1],f4[::1])",
    fastmath=True,
    cache=True,
    locals={
        "result": numba.types.float32,
        "diff": numba.types.float32,
        "dim": numba.types.intp,
        "i": numba.types.intp,
    },
)
def rdist(x, y):
    result = 0.0
    dim = x.shape[0]
    for i in range(dim):
        diff = x[i] - y[i]
        result += diff * diff

    return result


@numba.njit(inline="always")
def clip(val, lo, hi):
    if val > hi:
        return hi
    elif val < lo:
        return lo
    else:
        return val


@numba.njit(inline="always")
def get_range_limits(center, range_size, array_length):
    half_size = range_size // 2
    start = center - half_size

    # Handle boundary conditions
    if start < 0:
        start = 0
    elif start + range_size > array_length:
        start = max(0, array_length - range_size)

    return start


@numba.njit(
    [
        "void(f4[:, ::1], f4[:, ::1], i4[::1], i4[::1], i8, f8[::1], f8, f8, f8, i8, f8, f8[::1], f8[::1], f8[::1], i8, f4[:, ::1], f4[:, ::1], f4[:, ::1], f8, f8, i4[::1], i4[::1], i8, i8)",
        "void(f4[:, ::1], f4[:, ::1], i4[::1], i4[::1], i8, f4[::1], f8, f8, f8, i8, f8, f4[::1], f4[::1], f4[::1], i8, f4[:, ::1], f4[:, ::1], f4[:, ::1], f8, f8, i4[::1], i4[::1], i8, i8)",
    ],
    fastmath=True,
    parallel=True,
    locals={
        "updates": numba.types.float32[:, ::1],
        "from_node": numba.types.intp,
        "to_node": numba.types.intp,
        "raw_index": numba.types.intp,
        "dist_squared": numba.types.float32,
        "grad_coeff": numba.types.float32,
        "grad_d": numba.types.float32,
    },
)
def optimize_layout_euclidean_single_epoch_adam(
    head_embedding,
    tail_embedding,
    csr_indptr,
    csr_indices,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    gamma,
    dim,
    alpha,
    epochs_per_negative_sample,
    epoch_of_next_negative_sample,
    epoch_of_next_sample,
    n,
    updates,
    adam_m,
    adam_v,
    beta1,
    beta2,
    from_node_order,
    to_node_order,
    block_size=256,
    negative_selection_range=200_000,
):
    n_from_vertices = csr_indptr.shape[0] - 1
    negative_selection_range = max(200, min(n_vertices, negative_selection_range))
    negative_sample_scaling = negative_selection_range / n_vertices
    transform_mode = from_node_order.shape[0] != to_node_order.shape[0]
    for block_start in range(0, n_from_vertices, block_size):
        block_end = min(block_start + block_size, n_from_vertices)
        for raw_idx in numba.prange(block_start, block_end):
            node_idx = from_node_order[raw_idx]
            if transform_mode:
                from_node = node_idx
            else:
                from_node = to_node_order[node_idx]
            current = head_embedding[from_node]

            for raw_index in range(csr_indptr[from_node], csr_indptr[from_node + 1]):
                if epoch_of_next_sample[raw_index] <= n:
                    to_node = csr_indices[raw_index]
                    other = tail_embedding[to_node]

                    dist_squared = rdist(current, other)

                    if dist_squared > 0.0:
                        grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                        grad_coeff /= a * pow(dist_squared, b) + 1.0
                        for d in range(dim):
                            grad_d = grad_coeff * (current[d] - other[d])
                            updates[from_node, d] += grad_d

                    epoch_of_next_sample[raw_index] += epochs_per_sample[raw_index]

                    n_neg_samples = int(
                        (n - epoch_of_next_negative_sample[raw_index])
                        / epochs_per_negative_sample[raw_index]
                    )

                    for p in range(n_neg_samples):
                        to_node_raw_selection = (
                            raw_index * (n + p + 1)
                        ) % negative_selection_range
                        range_start = get_range_limits(
                            node_idx, negative_selection_range, n_vertices
                        )
                        to_node = to_node_order[
                            (range_start + to_node_raw_selection) % n_vertices
                        ]
                        other = tail_embedding[to_node]

                        dist_squared = rdist(current, other)

                        if dist_squared > 0.0:
                            grad_coeff = negative_sample_scaling * 2.0 * gamma * b
                            grad_coeff /= (0.001 + dist_squared) * (
                                a * pow(dist_squared, b) + 1
                            )

                            if grad_coeff > 0.0:
                                grad_norm = np.sqrt(
                                    grad_coeff * grad_coeff * dist_squared
                                )
                                scale = gamma * np.tanh(grad_norm / gamma) / grad_norm
                                for d in range(dim):
                                    updates[from_node, d] += (
                                        grad_coeff * (current[d] - other[d]) * scale
                                    )

                    epoch_of_next_negative_sample[raw_index] += (
                        n_neg_samples * epochs_per_negative_sample[raw_index]
                    )

        for node_idx in numba.prange(block_start, block_end):
            from_node = from_node_order[node_idx]
            for d in range(dim):
                if updates[from_node, d] != 0.0:
                    adam_m[from_node, d] = (
                        beta1 * adam_m[from_node, d]
                        + (1.0 - beta1) * updates[from_node, d]
                    )
                    adam_v[from_node, d] = (
                        beta2 * adam_v[from_node, d]
                        + (1.0 - beta2) * updates[from_node, d] ** 2
                    )
                    m_est = adam_m[from_node, d] / (1.0 - pow(beta1, n + 1))
                    v_est = adam_v[from_node, d] / (1.0 - pow(beta2, n + 1))
                    head_embedding[from_node, d] += (
                        alpha * m_est / (np.sqrt(v_est) + 1e-4)
                    )


@numba.njit(
    [
        "void(f4[:, ::1], f4[:, ::1], i4[::1], i4[::1], f4[::1], i8, f8[::1], f8, f8, f8, i8, f8, f8[::1], f8[::1], f8[::1], i8, f4[:, ::1], f4[:, ::1], f4[:, ::1], f8, f8, i4[::1], i4[::1])",
        "void(f4[:, ::1], f4[:, ::1], i4[::1], i4[::1], f4[::1], i8, f4[::1], f8, f8, f8, i8, f8, f4[::1], f4[::1], f4[::1], i8, f4[:, ::1], f4[:, ::1], f4[:, ::1], f8, f8, i4[::1], i4[::1])",
    ],
    fastmath=True,
    parallel=True,
    locals={
        "updates": numba.types.float32[:, ::1],
        "from_node": numba.types.intp,
        "to_node": numba.types.intp,
        "raw_index": numba.types.intp,
        "dist_squared": numba.types.float32,
        "grad_coeff": numba.types.float32,
        "grad_d": numba.types.float32,
    },
)
def optimize_small_layout_euclidean_single_epoch_adam(
    head_embedding,
    tail_embedding,
    csr_indptr,
    csr_indices,
    csr_data,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    gamma,
    dim,
    alpha,
    epochs_per_negative_sample,
    epoch_of_next_negative_sample,
    epoch_of_next_sample,
    n,
    updates,
    adam_m,
    adam_v,
    beta1,
    beta2,
    from_node_order,
    to_node_order,
):
    n_from_vertices = csr_indptr.shape[0] - 1
    transform_mode = from_node_order.shape[0] != to_node_order.shape[0]
    attraction_rescaling = 1.0 / pow(n + 1, 0.1)
    for raw_idx in numba.prange(n_from_vertices):
        node_idx = from_node_order[raw_idx]
        if transform_mode:
            from_node = node_idx
        else:
            from_node = to_node_order[node_idx]
        current = head_embedding[from_node]

        for raw_index in range(csr_indptr[from_node], csr_indptr[from_node + 1]):
            to_node = csr_indices[raw_index]
            other = tail_embedding[to_node]
            weight = csr_data[raw_index]

            dist_squared = rdist(current, other)

            if dist_squared > 0.0:
                grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                grad_coeff /= a * pow(dist_squared, b) + 1.0
                for d in range(dim):
                    grad_d = (
                        weight
                        * grad_coeff
                        * (current[d] - other[d])
                        * pow(dist_squared, attraction_rescaling)
                    )
                    updates[from_node, d] += grad_d

            for p in range(int(weight * 5)):
                to_node = (raw_index * (n + p + 1)) % n_vertices
                to_node_idx = np.where(
                    csr_indices[csr_indptr[from_node] : csr_indptr[from_node + 1]]
                    == to_node
                )[0]
                if to_node_idx.size == 0:
                    neg_weight = 1.0
                else:
                    neg_weight = 1.0 - csr_data[csr_indptr[from_node] + to_node_idx[0]]

                other = tail_embedding[to_node]

                dist_squared = rdist(current, other)

                if dist_squared > 0.0:
                    grad_coeff = 2.0 * gamma * b
                    grad_coeff /= (0.001 + dist_squared) * (
                        a * pow(dist_squared, b) + 1
                    )

                    if grad_coeff > 0.0:
                        grad_norm = np.sqrt(grad_coeff * grad_coeff * dist_squared)
                        scale = gamma * np.tanh(grad_norm / gamma) / grad_norm
                        for d in range(dim):
                            updates[from_node, d] += (
                                neg_weight
                                * grad_coeff
                                * (current[d] - other[d])
                                * scale
                            )

    for node_idx in numba.prange(n_from_vertices):
        from_node = from_node_order[node_idx]
        for d in range(dim):
            if updates[from_node, d] != 0.0:
                adam_m[from_node, d] = (
                    beta1 * adam_m[from_node, d] + (1.0 - beta1) * updates[from_node, d]
                )
                adam_v[from_node, d] = (
                    beta2 * adam_v[from_node, d]
                    + (1.0 - beta2) * updates[from_node, d] ** 2
                )
                m_est = adam_m[from_node, d] / (1.0 - pow(beta1, n + 1))
                v_est = adam_v[from_node, d] / (1.0 - pow(beta2, n + 1))
                head_embedding[from_node, d] += alpha * m_est / (np.sqrt(v_est) + 1e-4)


def _create_alpha_schedule(optimizer, n_epochs, initial_alpha, good_initialization):
    """Create alpha (learning rate) schedule based on optimizer and initialization quality."""
    if optimizer == "compatibility":
        return np.linspace(initial_alpha, 0.0, n_epochs, endpoint=False)

    elif optimizer in ["standard", "densmap_standard"]:
        if good_initialization:
            n_warm_up_epochs = int(max(200, n_epochs / 8))
            raw_alpha_schedule = np.asarray(
                [
                    (1.0 - (float(n) / float(n_warm_up_epochs))) ** 2
                    for n in range(n_warm_up_epochs)
                ]
                + [0.0] * (n_epochs - n_warm_up_epochs)
            )
            return 0.25 * raw_alpha_schedule * initial_alpha + 0.005
        else:
            raw_alpha_schedule = np.asarray(
                [
                    0.25 * (1.0 - (float(n) / float(n_epochs))) ** 2
                    for n in range(n_epochs)
                ]
            )
            return raw_alpha_schedule * initial_alpha

    elif optimizer in ["adam", "densmap_adam"]:
        if good_initialization:
            n_warm_up_epochs = int(min(100, n_epochs / 8))
        else:
            n_warm_up_epochs = int(min(n_epochs / 2, 100))

        if good_initialization:
            return np.concatenate(
                [
                    [
                        (0.5 * initial_alpha - 0.1)
                        * (1.0 - (float(n) / float(n_warm_up_epochs))) ** 2
                        + 0.1
                        for n in range(n_warm_up_epochs)
                    ],
                    [
                        0.15
                        * (
                            1.0
                            - (
                                float(n - n_warm_up_epochs)
                                / float(n_epochs - n_warm_up_epochs)
                            )
                        )
                        + 0.05
                        for n in range(n_warm_up_epochs, n_epochs)
                    ],
                ]
            )
        else:
            return np.concatenate(
                [
                    [
                        (2.0 * initial_alpha - 0.1)
                        * (1.0 - (float(n) / float(n_warm_up_epochs))) ** 2
                        + 0.1
                        for n in range(n_warm_up_epochs)
                    ],
                    [
                        0.15
                        * (
                            1.0
                            - (
                                float(n - n_warm_up_epochs)
                                / float(n_epochs - n_warm_up_epochs)
                            )
                        )
                        + 0.05
                        for n in range(n_warm_up_epochs, n_epochs)
                    ],
                ]
            )


def _create_momentum_schedule(optimizer, n_epochs, good_initialization):
    """Create momentum schedule based on optimizer and initialization quality."""
    if optimizer in ["adam", "densmap_adam"]:
        return np.zeros(n_epochs, dtype=np.float32)

    elif optimizer in ["standard", "densmap_standard"]:
        if good_initialization:
            n_warm_up_epochs = int(max(200, n_epochs / 8))
            raw_alpha_schedule = np.asarray(
                [
                    (1.0 - (float(n) / float(n_warm_up_epochs))) ** 2
                    for n in range(n_warm_up_epochs)
                ]
                + [0.0] * (n_epochs - n_warm_up_epochs)
            )
            return np.asarray(
                [0.5 * (1.0 - raw_alpha_schedule[n]) for n in range(n_warm_up_epochs)]
                + [0.5] * (n_epochs - n_warm_up_epochs)
            )
        else:
            raw_alpha_schedule = np.asarray(
                [
                    0.25 * (1.0 - (float(n) / float(n_epochs))) ** 2
                    for n in range(n_epochs)
                ]
            )
            return np.asarray(
                [0.5 * (1.0 - raw_alpha_schedule[n]) for n in range(n_epochs)]
            )

    return np.zeros(n_epochs, dtype=np.float32)


def _create_adam_schedules(
    optimizer,
    n_epochs,
    good_initialization,
    gamma,
    n_vertices,
    negative_selection_range,
):
    """Create beta1, beta2, and gamma schedules for Adam optimizer."""
    if optimizer not in ["adam", "densmap_adam", "standard"]:
        return None, None, None, None

    if good_initialization:
        n_warm_up_epochs = int(min(100, n_epochs / 4))
    else:
        n_warm_up_epochs = int(min(n_epochs // 2, 100))  # Use n_epochs/2 but cap at 100

    beta1_schedule = np.concatenate(
        [
            [
                0.2 + (0.7 * (float(n) / float(n_warm_up_epochs)))
                for n in range(n_warm_up_epochs)
            ],
            np.full(n_epochs - n_warm_up_epochs, 0.9),
        ]
    )

    beta2_schedule = np.concatenate(
        [
            [
                0.79 + (0.2 * ((float(n) / float(n_warm_up_epochs))))
                for n in range(n_warm_up_epochs)
            ],
            np.full(n_epochs - n_warm_up_epochs, 0.99),
        ]
    )

    if good_initialization:
        gamma_schedule = (
            np.concatenate(
                [
                    [
                        1.5 * np.sqrt(float(n) / float(n_warm_up_epochs))
                        for n in range(n_warm_up_epochs)
                    ],
                    [
                        0.5
                        * (
                            1.0
                            - (
                                float(n - n_warm_up_epochs)
                                / float(n_epochs - n_warm_up_epochs)
                            )
                        )
                        + 1.0
                        for n in range(n_warm_up_epochs, n_epochs)
                    ],
                ]
            )
            * gamma
            * max(np.sqrt(n_epochs / 100.0), 1.0)
        )
        # gamma_schedule = np.full(n_epochs, gamma * max(np.sqrt(n_epochs / 100.0), 1.0), dtype=np.float32)
    else:
        gamma_schedule = (
            np.concatenate(
                [
                    [
                        3.0 * np.sqrt(float(n) / float(n_warm_up_epochs))
                        for n in range(n_warm_up_epochs)
                    ],
                    [
                        1.0
                        * (
                            1.0
                            - float(n - n_warm_up_epochs)
                            / float(n_epochs - n_warm_up_epochs)
                        )
                        + 2.0
                        for n in range(n_warm_up_epochs, n_epochs)
                    ],
                ]
            )
            * gamma
            * max(np.sqrt(n_epochs / 100.0), 1.0)
        )
        # gamma_schedule = np.full(n_epochs, gamma * max(np.sqrt(n_epochs / 100.0), 1.0), dtype=np.float32)

    negative_selection_range_schedule = np.linspace(
        n_vertices,
        negative_selection_range,
        n_epochs,
        dtype=np.int32,
    )
    # negative_selection_range_schedule = np.full(
    #     n_epochs, negative_selection_range, dtype=np.int32
    # )

    return (
        beta1_schedule,
        beta2_schedule,
        gamma_schedule,
        negative_selection_range_schedule,
    )


def optimize_layout_euclidean(
    head_embedding,
    tail_embedding,
    head,
    tail,
    n_epochs,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state,
    gamma=1.0,
    initial_alpha=1.0,
    negative_sample_rate=5.0,
    parallel=False,
    verbose=False,
    densmap_kwds=None,
    tqdm_kwds=None,
    move_other=False,
    csr_indptr=None,
    csr_indices=None,
    csr_data=None,
    optimizer="adam",
    good_initialization=False,
    random_state=None,
    negative_selection_range=200_000,
):
    dim = head_embedding.shape[1]
    if random_state is None:
        random_state = np.random.RandomState()
    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()
    updates = np.zeros((head_embedding.shape[0], dim), dtype=np.float32)
    node_order = np.arange(head_embedding.shape[0], dtype=np.int32)
    if tail_embedding.shape[0] != head_embedding.shape[0] or optimizer == "adam":
        to_node_order = np.arange(tail_embedding.shape[0], dtype=np.int32)
    else:
        to_node_order = node_order
    block_size = 4096

    # Create learning schedules
    alpha_schedule = _create_alpha_schedule(
        optimizer, n_epochs, initial_alpha, good_initialization
    )
    momentum_schedule = _create_momentum_schedule(
        optimizer, n_epochs, good_initialization
    )

    # Adam-specific schedules
    (
        beta1_schedule,
        beta2_schedule,
        gamma_schedule,
        negative_selection_range_schedule,
    ) = _create_adam_schedules(
        optimizer,
        n_epochs,
        good_initialization,
        gamma,
        n_vertices,
        negative_selection_range,
    )

    # Adjust negative sampling rates for non-compatibility optimizers
    if optimizer != "compatibility":
        epochs_per_negative_sample *= 1.5
        epoch_of_next_negative_sample *= 1.5

    densmap = False
    adam_m = np.zeros_like(updates)
    adam_v = np.zeros_like(updates)

    if densmap_kwds is None:
        densmap_kwds = {}
    if tqdm_kwds is None:
        tqdm_kwds = {}

    if "disable" not in tqdm_kwds:
        tqdm_kwds["disable"] = not verbose

    embeddings = []
    b_schedule = np.linspace(1.0, b, n_epochs)

    for n in tqdm(range(n_epochs), **tqdm_kwds):
        if head_embedding.shape[0] <= 256:
            optimize_small_layout_euclidean_single_epoch_adam(
                head_embedding,
                tail_embedding,
                csr_indptr,
                csr_indices,
                csr_data,
                n_vertices,
                epochs_per_sample,
                a,
                b_schedule[n],
                gamma_schedule[n],
                dim,
                alpha_schedule[n],
                epochs_per_negative_sample,
                epoch_of_next_negative_sample,
                epoch_of_next_sample,
                n,
                updates,
                adam_m,
                adam_v,
                beta1_schedule[n],
                beta2_schedule[n],
                node_order,
                to_node_order,
            )
            updates[:] = 0.0
            random_state.shuffle(node_order)
            random_state.shuffle(to_node_order)
        else:
            optimize_layout_euclidean_single_epoch_adam(
                head_embedding,
                tail_embedding,
                csr_indptr,
                csr_indices,
                n_vertices,
                epochs_per_sample,
                a,
                b_schedule[n],
                gamma_schedule[n],
                dim,
                alpha_schedule[n],
                epochs_per_negative_sample,
                epoch_of_next_negative_sample,
                epoch_of_next_sample,
                n,
                updates,
                adam_m,
                adam_v,
                beta1_schedule[n],
                beta2_schedule[n],
                node_order,
                to_node_order,
                block_size=n_vertices,
                negative_selection_range=negative_selection_range_schedule[n],
            )
            updates[:] = 0.0
            random_state.shuffle(node_order)
            if negative_selection_range_schedule[n] < n_vertices:
                projection_direction = random_state.randn(dim)
                projection_direction /= np.linalg.norm(projection_direction)
                projection = np.dot(head_embedding, projection_direction)
                to_node_order = np.argsort(projection).astype(np.int32)
            else:
                random_state.shuffle(to_node_order)

        embeddings.append(head_embedding.copy())

    return head_embedding, embeddings


def umap_label_propagation_init(
    graph,
    data,
    subset_mask,
    a,
    b,
    n_iter=100,
    n_embedding_epochs=32,
    approx_n_parts=None,
    n_components=2,
    scaling=1.0,
    random_scale=1.0,
    random_state=None,
    recursive_init=True,
    base_init_threshold=256,
    expansion_alpha=0.25,
    depth=1,
    verbose=False,
):

    if random_state is None:
        random_state = np.random.RandomState()

    if graph.shape[0] <= base_init_threshold:
        result = data
        # Recenter
        scale = (
            np.log10(result.shape[0]) * 3 * (np.log2(depth + 1))
        )  # Added log2(gamma) to scale with repulsion strength
        result -= np.mean(result, 0)
        result *= scale / (np.quantile(result, 0.95, 0) - np.quantile(result, 0.05, 0))

        # fig, ax = plt.subplots(figsize=(8,8))
        # ax.scatter(*result.T, s=depth / 10, c=target, cmap="Spectral")

        return result.astype(np.float32, order="C"), [result.copy()]

    if approx_n_parts is None:
        approx_n_parts = max(base_init_threshold, int(graph.shape[0] // 4))

    # Ensure we have fewer parts than samples
    approx_n_parts = min(approx_n_parts, graph.shape[0] // 2)
    if approx_n_parts < 2:
        approx_n_parts = 2

    # Initialize the label propagation process
    rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)
    labels = np.full(graph.shape[0], -1, dtype=np.int32)
    # labels = umap_initialize_labels(labels, approx_n_parts, rng_state)
    labels = umap_initialize_labels_from_hubs(
        labels, approx_n_parts, np.squeeze(np.array(graph.sum(axis=0)))
    )
    # labels = evoc_initialize_labels(labels, approx_n_parts, random_state)
    # Perform the label propagation iterations
    prev_unlabeled = np.sum(labels < 0)
    for i in range(n_iter):
        labels = umap_label_prop_iteration(
            graph.indptr,
            graph.indices,
            graph.data,
            labels,
            rng_state,
        )
        if i % 5 == 0:
            unlabeled = np.sum(labels < 0)
            if unlabeled == 0 or unlabeled == prev_unlabeled:
                print(f"breaking after {i} iterations")
                break
        prev_unlabeled = unlabeled
    # Handle outliers
    labels = umap_label_outliers(
        graph.indptr,
        graph.indices,
        labels,
        rng_state,
    )
    # Remap labels to a contiguous range
    labels = umap_remap_labels(labels)

    base_reduction_map = csr_matrix(
        (np.ones(labels.shape[0]), labels, np.arange(labels.shape[0] + 1)),
        shape=(labels.shape[0], labels.max() + 1),
    )
    complement_graph = graph.astype(np.float64)
    complement_graph.data = np.log1p(-np.clip(complement_graph.data, 0.0, 1.0 - 1e-16))
    reduced_graph = base_reduction_map.T * complement_graph * base_reduction_map
    reduced_graph.data = 1.0 - np.exp(reduced_graph.data)
    reduced_graph.eliminate_zeros()
    reduced_graph = reduced_graph.astype(np.float32)

    if not np.all(subset_mask):
        subset_reduction_map = normalize(
            base_reduction_map[subset_mask], axis=0, norm="l1"
        )
        reduced_data = subset_reduction_map.T * data
    else:
        l1_normalized_reduction_map = normalize(base_reduction_map, axis=0, norm="l1")
        reduced_data = l1_normalized_reduction_map.T * data

    if recursive_init:
        reduced_init, reduced_steps = umap_label_propagation_init(
            reduced_graph,
            reduced_data,
            np.ones(reduced_graph.shape[0], dtype=np.bool_),
            a,  # * 0.75,  # Note, changing initialization strategy TODO: check this works
            b,
            n_iter=n_iter,
            approx_n_parts=approx_n_parts
            // 4,  # max(base_init_threshold, int(np.sqrt(graph.shape[0] * base_init_threshold))),
            n_embedding_epochs=int(
                n_embedding_epochs * np.pow(2, 0.25)
            ),  # int(n_embedding_epochs * np.sqrt(2)),
            n_components=n_components,
            scaling=scaling,
            random_scale=random_scale,
            random_state=random_state,
            recursive_init=True,
            base_init_threshold=base_init_threshold,
            depth=depth + 1,
            verbose=verbose,
            expansion_alpha=np.clip(expansion_alpha * 2, 0, 1),
        )
        reduced_init = reduced_init.astype(np.float32)
        good_initialization = approx_n_parts // 4 > base_init_threshold
    else:
        reduced_init = None
        good_initialization = False

    epochs_per_sample = make_epochs_per_sample(reduced_graph.data, n_embedding_epochs)
    reduced_layout, steps = optimize_layout_euclidean(
        reduced_init,
        reduced_init,
        None,
        None,
        n_embedding_epochs,
        reduced_graph.shape[0],
        epochs_per_sample,
        a,
        b,
        rng_state,
        8.0,  # 1.5,
        0.5,
        1,
        parallel=True,
        verbose=True,
        densmap_kwds={},
        tqdm_kwds={"desc": f"Init recursion depth {depth}", "position": 1},
        move_other=False,
        csr_indptr=reduced_graph.indptr,
        csr_indices=reduced_graph.indices,
        csr_data=reduced_graph.data,
        random_state=random_state,
        optimizer="adam",
        good_initialization=True,
        negative_selection_range=reduced_init.shape[0] / 2,
    )

    data_expander = normalize(graph @ base_reduction_map, norm="l1")
    result = (
        expansion_alpha * data_expander @ reduced_layout
        + (1.0 - expansion_alpha)
        * normalize(base_reduction_map, norm="l1")
        @ reduced_layout
    )

    steps = reduced_steps + steps
    for i, step in enumerate(steps):
        steps[i] = normalize(base_reduction_map, norm="l1") @ step
        steps[i] -= np.mean(steps[i], 0)
        steps[i] *= (
            scaling / (np.quantile(steps[i], 0.95, 0) - np.quantile(steps[i], 0.05, 0))
        ).astype(np.float32, order="C")

    result -= np.mean(result, 0)
    result *= (
        scaling / (np.quantile(result, 0.95, 0) - np.quantile(result, 0.05, 0))
    ).astype(np.float32, order="C")
    # result = (scaling * (result - result.mean(axis=0))).astype(np.float32)

    unexpanded = steps[-1]
    for alpha in np.linspace(0, 1, 16):
        steps.append(alpha * result + (1.0 - alpha) * unexpanded)

    prerotation = result.copy()
    # Procrustes
    rotation = procrustes_align(data, result[subset_mask])
    result = result @ rotation

    for alpha in np.linspace(0, 1, 8):
        steps.append(alpha * result + (1.0 - alpha) * prerotation)

    return result.astype(np.float32), steps


def recursive_init_umap(graph, data):
    n = data.shape[0]
    sample_size = min(16384, n)

    scale = np.log10(data.shape[0]) * 3 * (np.log2(2 + 1))

    if sample_size < n:
        sample = np.sort(np.random.choice(n, size=sample_size, replace=False))
        pca_sample_mask = np.zeros(n, dtype=np.bool_)
        pca_sample_mask[sample] = True
        data_sample = data[sample]
    else:
        pca_sample_mask = np.ones(n, dtype=np.bool_)
        data_sample = data

    X = data_sample - data_sample.mean(axis=0)
    U, S, _ = randomized_svd(
        X, n_components=2, n_iter=1, n_oversamples=8, random_state=42
    )

    pca = (U * S).astype(np.float32, order="C")

    pca -= pca.mean(axis=0)
    pca /= pca.max(axis=0) - pca.min(axis=0)
    pca *= scale
    init, init_steps = umap_label_propagation_init(
        graph,
        pca,
        pca_sample_mask,
        a=1.0,
        b=1.0,
        base_init_threshold=1024,
        n_embedding_epochs=64,
        approx_n_parts=16384,
        scaling=scale,
    )
    return init, init_steps


def make_umap_animation_data(
    graph,
    data,
    a=1.0,
    b=0.75,
    gamma=4.0,
    negative_selection_range_denominator=2,
    init_scale=1.0,
):
    umap_init, init_steps = recursive_init_umap(graph, data)
    init_steps = [step * init_scale for step in init_steps]
    umap_init *= init_scale
    epochs_per_sample = make_epochs_per_sample(graph.data, 500)
    rng_state = np.random.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)
    final_layout, steps = optimize_layout_euclidean(
        umap_init,
        umap_init,
        None,
        None,
        500,
        graph.shape[0],
        epochs_per_sample,
        a,
        b,
        rng_state,
        gamma,  # 1.5,
        0.5,
        1,
        parallel=True,
        verbose=True,
        densmap_kwds={},
        tqdm_kwds={"desc": f"Init recursion depth {0}", "position": 0},
        move_other=False,
        csr_indptr=graph.indptr,
        csr_indices=graph.indices,
        csr_data=graph.data,
        random_state=np.random,
        optimizer="adam",
        good_initialization=True,
        negative_selection_range=umap_init.shape[0]
        / negative_selection_range_denominator,
    )
    init_steps += steps
    return init_steps


mnist = sklearn.datasets.fetch_openml("mnist_784")
mnist_data = np.ascontiguousarray(mnist.data, dtype=np.float32)
mnist_target = mnist.target.astype(np.uint8)

fmnist = sklearn.datasets.fetch_openml("Fashion-MNIST")
fmnist_data = np.ascontiguousarray(fmnist.data, dtype=np.float32)
fmnist_target = fmnist.target.astype(np.uint8)


nn_inds, nn_dists = evoc.knn_graph.knn_graph(
    mnist_data, n_neighbors=15, random_state=None
)
graph = evoc.graph_construction.neighbor_graph_matrix(15, nn_inds, nn_dists, True)

nn_inds_f, nn_dists_f = evoc.knn_graph.knn_graph(
    fmnist_data, n_neighbors=15, random_state=None
)
graph_f = evoc.graph_construction.neighbor_graph_matrix(15, nn_inds_f, nn_dists_f, True)

anim_steps = make_umap_animation_data(graph, mnist_data)
np.save("mnist_umap_animation_steps_1.npy", anim_steps)
np.save("mnist_targets.npy", mnist_target)

anim_steps = make_umap_animation_data(graph_f, fmnist_data)
np.save("fashion_mnist_umap_animation_steps_1.npy", anim_steps)
np.save("fashion_mnist_targets.npy", fmnist_target)

anim_steps = make_umap_animation_data(graph, mnist_data, a=0.85, b=0.55, gamma=8.0)
np.save("mnist_umap_animation_steps_2.npy", anim_steps)


anim_steps = make_umap_animation_data(graph_f, fmnist_data, a=0.85, b=0.55, gamma=8.0)
np.save("fashion_mnist_umap_animation_steps_2.npy", anim_steps)


anim_steps = make_umap_animation_data(
    graph,
    mnist_data,
    a=0.6,
    b=0.4,
    gamma=16.0,
    negative_selection_range_denominator=128,
    init_scale=2.0,
)
np.save("mnist_umap_animation_steps_3.npy", anim_steps)


anim_steps = make_umap_animation_data(
    graph_f,
    fmnist_data,
    a=0.6,
    b=0.4,
    gamma=16.0,
    negative_selection_range_denominator=128,
    init_scale=2.0,
)
np.save("fashion_mnist_umap_animation_steps_3.npy", anim_steps)
