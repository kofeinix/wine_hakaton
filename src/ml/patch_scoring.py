from __future__ import annotations

import numpy as np


def patch_similarity_score(
    query_patches: list[list[float]] | np.ndarray,
    candidate_patches: list[list[float]] | np.ndarray,
    query_grid: tuple[int, int] | None,
    candidate_grid: tuple[int, int] | None,
    top_fraction: float = 0.20,
    strategy: str = "spatial",
) -> float:
    query = np.asarray(query_patches, dtype=np.float32)
    candidate = np.asarray(candidate_patches, dtype=np.float32)
    if query.size == 0 or candidate.size == 0:
        return 0.0

    if query.ndim != 2 or candidate.ndim != 2 or query.shape[1] != candidate.shape[1]:
        return 0.0

    sim = query @ candidate.T
    q_to_c = sim.max(axis=1)
    c_to_q = sim.max(axis=0)
    q_match = sim.argmax(axis=1)

    if strategy == "spatial":
        return _spatial_consistency(q_match, q_to_c, query_grid, candidate_grid, top_fraction)
    if strategy == "sym10":
        return 0.5 * (_top_fraction_mean(q_to_c, 0.10) + _top_fraction_mean(c_to_q, 0.10))
    if strategy == "sym20":
        return 0.5 * (_top_fraction_mean(q_to_c, 0.20) + _top_fraction_mean(c_to_q, 0.20))
    if strategy == "q_top20":
        return _top_fraction_mean(q_to_c, 0.20)
    if strategy == "c_top20":
        return _top_fraction_mean(c_to_q, 0.20)

    symmetric_top20 = 0.5 * (
        _top_fraction_mean(q_to_c, top_fraction)
        + _top_fraction_mean(c_to_q, top_fraction)
    )
    coverage = 0.5 * (float(q_to_c.mean()) + float(c_to_q.mean()))
    spatial = _spatial_consistency(q_match, q_to_c, query_grid, candidate_grid, top_fraction)
    if strategy == "sym80_spatial20":
        return float(0.8 * symmetric_top20 + 0.2 * spatial)
    if strategy == "sym80_cov20":
        return float(0.8 * symmetric_top20 + 0.2 * coverage)
    if strategy == "cov50_spatial50":
        return float(0.5 * coverage + 0.5 * spatial)
    if strategy == "sym70_cov15_spatial15":
        return float(0.7 * symmetric_top20 + 0.15 * coverage + 0.15 * spatial)
    if strategy == "sym80_cov10_spatial10":
        return float(0.8 * symmetric_top20 + 0.1 * coverage + 0.1 * spatial)
    return float(0.6 * symmetric_top20 + 0.2 * coverage + 0.2 * spatial)


def _top_fraction_mean(values: np.ndarray, fraction: float) -> float:
    if values.size == 0:
        return 0.0
    k = max(1, int(values.size * fraction))
    indices = np.argpartition(values, -k)[-k:]
    return float(values[indices].mean())


def _spatial_consistency(
    q_match: np.ndarray,
    q_to_c: np.ndarray,
    query_grid: tuple[int, int] | None,
    candidate_grid: tuple[int, int] | None,
    top_fraction: float,
) -> float:
    if query_grid is None or candidate_grid is None:
        return 0.0
    qh, qw = query_grid
    ch, cw = candidate_grid
    if qh <= 1 or qw <= 1 or ch <= 1 or cw <= 1:
        return 0.0

    k = max(1, int(q_to_c.size * top_fraction))
    q_indices = np.argpartition(q_to_c, -k)[-k:]
    c_indices = q_match[q_indices]

    q_y = q_indices // qw
    q_x = q_indices % qw
    c_y = c_indices // cw
    c_x = c_indices % cw

    q_coords = np.stack([q_y / (qh - 1), q_x / (qw - 1)], axis=1)
    c_coords = np.stack([c_y / (ch - 1), c_x / (cw - 1)], axis=1)
    distance = np.linalg.norm(q_coords - c_coords, axis=1)
    return float(np.exp(-4.0 * distance).mean())
