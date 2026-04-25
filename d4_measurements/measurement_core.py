"""Core D4 measurement geometry, operator construction, and estimators."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Callable

import numpy as np

try:
    from .deterministic_geometry import (
        COLOR_NAMES,
        Geometry,
        Spec,
        flip_samples,
        honeycomb_node_color,
        paper_magnetic_string,
        phase,
        representative_open_paths,
        same_torus_node_path_flips,
        star_ring_pairs,
        triangle_anticommutators,
    )
except ImportError:  # pragma: no cover - supports direct script-style imports.
    from deterministic_geometry import (
        COLOR_NAMES,
        Geometry,
        Spec,
        flip_samples,
        honeycomb_node_color,
        paper_magnetic_string,
        phase,
        representative_open_paths,
        same_torus_node_path_flips,
        star_ring_pairs,
        triangle_anticommutators,
    )


COLORS = (0, 1, 2)


@dataclass(frozen=True)
class PathSpec:
    color: int
    label: str
    sites: tuple[int, ...]
    endpoints: tuple[Any, ...]
    distance: int
    weight: float


@dataclass(frozen=True)
class MagneticSpec:
    color: int
    label: str
    spec: Spec
    start: tuple[str, int, int]
    end: tuple[str, int, int]
    path_nodes: tuple[tuple[str, int, int], ...]
    passed_sites: tuple[int, ...]
    skipped_stars: tuple[int, ...]
    dressing_kind: str
    weight: float


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def color_plaquettes(geom: Geometry, color: int) -> np.ndarray:
    return np.flatnonzero(geom.plaquette_color_labels == color)


def triangle_sets_by_color(geom: Geometry) -> dict[int, list[tuple[str, int, np.ndarray]]]:
    out = {color: [] for color in COLORS}
    for orient, p, tri in geom.triangles():
        color = int(geom.site_color_labels[int(tri[0])])
        out[color].append((orient, p, tri))
    return out


def electric_graph(geom: Geometry, color: int) -> dict[int, list[tuple[int, int]]]:
    plaquettes = set(map(int, color_plaquettes(geom, color)))
    site_to_plaq: dict[int, list[int]] = {}
    for p in plaquettes:
        for site in geom.x_list[p]:
            site_to_plaq.setdefault(int(site), []).append(int(p))
    graph = {p: [] for p in plaquettes}
    for site, endpoints in site_to_plaq.items():
        endpoints = [p for p in endpoints if p in plaquettes]
        if len(endpoints) != 2:
            raise ValueError(("bad electric edge", color, site, endpoints))
        a, b = endpoints
        graph[a].append((b, int(site)))
        graph[b].append((a, int(site)))
    return graph


def shortest_path_sites(graph: dict[int, list[tuple[int, int]]], start: int, end: int) -> tuple[int, ...] | None:
    if start == end:
        return ()
    queue = [start]
    parent: dict[int, tuple[int, int] | None] = {start: None}
    for node in queue:
        for nxt, site in graph[node]:
            if nxt in parent:
                continue
            parent[nxt] = (node, site)
            if nxt == end:
                queue = []
                break
            queue.append(nxt)
    if end not in parent:
        return None
    sites = []
    current = end
    while current != start:
        prev_entry = parent[current]
        if prev_entry is None:
            raise AssertionError("empty parent on nontrivial path")
        prev, site = prev_entry
        sites.append(site)
        current = prev
    return tuple(reversed(sites))


def build_electric_specs(
    geom: Geometry,
    max_pairs_per_color: int | None,
    string_set: str,
) -> dict[int, list[PathSpec]]:
    out: dict[int, list[PathSpec]] = {}
    for color in COLORS:
        graph = electric_graph(geom, color)
        nodes = sorted(graph)
        candidates = []
        for i, start in enumerate(nodes):
            for end in nodes[i + 1 :]:
                sites = shortest_path_sites(graph, start, end)
                if sites is None or not sites:
                    continue
                candidates.append((len(sites), start, end, sites))
        candidates.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))

        if string_set == "paper":
            if max_pairs_per_color is not None:
                candidates = candidates[:max_pairs_per_color]
            specs = [
                PathSpec(
                    color=color,
                    label=f"{COLOR_NAMES[color]}_e_self",
                    sites=(),
                    endpoints=("self", int(len(nodes))),
                    distance=0,
                    weight=float(len(nodes)),
                )
            ]
            specs.extend(
                PathSpec(
                    color=color,
                    label=f"{COLOR_NAMES[color]}_e_{idx:03d}",
                    sites=tuple(int(x) for x in sites),
                    endpoints=(int(start), int(end)),
                    distance=int(distance),
                    weight=2.0,
                )
                for idx, (distance, start, end, sites) in enumerate(candidates)
            )
            out[color] = specs
            continue

        if max_pairs_per_color is None:
            raise ValueError("representative electric strings require --max-e-pairs-per-color")
        chosen = candidates[:max_pairs_per_color]
        out[color] = [
            PathSpec(
                color=color,
                label=f"{COLOR_NAMES[color]}_e_{idx:03d}",
                sites=tuple(int(x) for x in sites),
                endpoints=(int(start), int(end)),
                distance=int(distance),
                weight=1.0,
            )
            for idx, (distance, start, end, sites) in enumerate(chosen)
        ]
    return out


def honeycomb_nodes(geom: Geometry, color: int, orient: str | None = None) -> list[tuple[str, int, int]]:
    nodes = [
        (node_orient, x, y)
        for y in range(geom.L)
        for x in range(geom.L)
        for node_orient in ("L", "R")
        if (orient is None or node_orient == orient)
        and honeycomb_node_color((node_orient, x, y)) == color
    ]
    return sorted(nodes)


def magnetic_candidates(
    geom: Geometry,
    color: int,
    string_set: str,
) -> list[tuple[tuple[str, int, int], tuple[str, int, int], tuple[int, ...]]]:
    if string_set == "representative":
        raise ValueError("representative magnetic strings are built directly from representative_open_paths")

    candidates = []
    for start in honeycomb_nodes(geom, color, orient="L"):
        for end in honeycomb_nodes(geom, color, orient="R"):
            flips = same_torus_node_path_flips(geom, color, start, end)
            endpoints = triangle_anticommutators(geom, Spec(flips=flips, cz_pairs=()))
            if len(endpoints) != 2:
                raise ValueError(("bad bare magnetic endpoints", color, start, end, endpoints))
            candidates.append((start, end, flips))
    return sorted(candidates, key=lambda item: (-len(item[2]), item[0], item[1]))


def build_magnetic_specs(
    geom: Geometry,
    max_pairs_per_color: int | None,
    string_set: str,
) -> dict[int, list[MagneticSpec]]:
    out: dict[int, list[MagneticSpec]] = {}
    for color in COLORS:
        specs = []
        if string_set == "representative":
            if max_pairs_per_color is None:
                raise ValueError("representative magnetic strings require --max-m-pairs-per-color")
            candidates = representative_open_paths(geom, color, count=max_pairs_per_color)
        else:
            candidates = magnetic_candidates(geom, color, string_set)
            if max_pairs_per_color is not None:
                candidates = candidates[:max_pairs_per_color]

        for idx, (start, end, flips) in enumerate(candidates):
            endpoints = triangle_anticommutators(geom, Spec(flips=flips, cz_pairs=()))
            skipped = tuple(sorted(int(endpoint[1]) for endpoint in endpoints))
            paper_string = paper_magnetic_string(geom, color, start, end)
            if set(paper_string.spec.flips) != set(flips):
                raise ValueError(
                    (
                        "paper W_m path support mismatch",
                        color,
                        start,
                        end,
                        flips,
                        paper_string.spec.flips,
                    )
                )
            specs.append(
                MagneticSpec(
                    color=color,
                    label=f"{COLOR_NAMES[color]}_m_{idx:03d}",
                    spec=paper_string.spec,
                    start=start,
                    end=end,
                    path_nodes=paper_string.path_nodes,
                    passed_sites=paper_string.passed_sites,
                    skipped_stars=skipped,
                    dressing_kind="paper_closed_form",
                    weight=1.0,
                )
            )
        out[color] = specs
    return out


def specs_to_manifest(e_specs: dict[int, list[PathSpec]], m_specs: dict[int, list[MagneticSpec]]) -> dict[str, Any]:
    return {
        "electric": {
            COLOR_NAMES[color]: [
                {
                    "label": spec.label,
                    "sites": list(spec.sites),
                    "endpoints": list(spec.endpoints),
                    "distance": spec.distance,
                    "weight": spec.weight,
                }
                for spec in specs
            ]
            for color, specs in e_specs.items()
        },
        "magnetic": {
            COLOR_NAMES[color]: [
                {
                    "label": spec.label,
                    "flips": list(spec.spec.flips),
                    "cz_pairs": [list(pair) for pair in spec.spec.cz_pairs],
                    "start": list(spec.start),
                    "end": list(spec.end),
                    "path_nodes": [list(node) for node in spec.path_nodes],
                    "passed_sites": list(spec.passed_sites),
                    "skipped_stars": list(spec.skipped_stars),
                    "dressing_kind": spec.dressing_kind,
                    "weight": spec.weight,
                }
                for spec in specs
            ]
            for color, specs in m_specs.items()
        },
    }


def operator_config_hash(operator_manifest: dict[str, Any]) -> str:
    blob = json.dumps(operator_manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def star_local_values(
    log_value_fn: Callable[[np.ndarray], np.ndarray],
    geom: Geometry,
    samples: np.ndarray,
    base_log: np.ndarray,
) -> np.ndarray:
    values = np.zeros((samples.shape[0], 3), dtype=np.complex128)
    counts = np.zeros(3, dtype=np.int64)
    for p in range(geom.N_plaquette):
        star = Spec(
            flips=tuple(int(x) for x in geom.x_list[p]),
            cz_pairs=star_ring_pairs(geom, p),
        )
        ph = phase(samples, star.cz_pairs).astype(np.complex128)
        flipped = flip_samples(samples, star.flips)
        log_flipped = log_value_fn(flipped)
        local = ph * np.exp(log_flipped - base_log)
        color = int(geom.plaquette_color_labels[p])
        values[:, color] += local
        counts[color] += 1
    return values / counts[np.newaxis, :]


def triangle_local_values(geom: Geometry, samples: np.ndarray) -> np.ndarray:
    values = np.zeros((samples.shape[0], 3), dtype=np.float64)
    counts = np.zeros(3, dtype=np.int64)
    for color, triangles in triangle_sets_by_color(geom).items():
        for _, _, tri in triangles:
            values[:, color] += np.prod(samples[:, tri], axis=1)
            counts[color] += 1
    return values / counts[np.newaxis, :]


def electric_local_values(samples: np.ndarray, e_specs: dict[int, list[PathSpec]]) -> np.ndarray:
    values = np.zeros((samples.shape[0], 3), dtype=np.float64)
    weights = np.zeros(3, dtype=np.float64)
    for color, specs in e_specs.items():
        for spec in specs:
            if spec.sites:
                local = np.prod(samples[:, list(spec.sites)], axis=1)
            else:
                local = np.ones(samples.shape[0], dtype=np.float64)
            values[:, color] += spec.weight * local
            weights[color] += spec.weight
    return values / np.maximum(weights, 1.0)[np.newaxis, :]


def magnetic_local_values(
    log_value_fn: Callable[[np.ndarray], np.ndarray],
    samples: np.ndarray,
    base_log: np.ndarray,
    m_specs: dict[int, list[MagneticSpec]],
) -> np.ndarray:
    values = np.zeros((samples.shape[0], 3), dtype=np.complex128)
    weights = np.zeros(3, dtype=np.float64)
    for color, specs in m_specs.items():
        for m_spec in specs:
            spec = m_spec.spec
            ph = phase(samples, spec.cz_pairs).astype(np.complex128)
            flipped = flip_samples(samples, spec.flips)
            log_flipped = log_value_fn(flipped)
            values[:, color] += m_spec.weight * ph * np.exp(log_flipped - base_log)
            weights[color] += m_spec.weight
    return values / np.maximum(weights, 1.0)[np.newaxis, :]


def measure_observable_batch(
    samples: np.ndarray,
    log_value_fn: Callable[[np.ndarray], np.ndarray],
    geom: Geometry,
    e_specs: dict[int, list[PathSpec]],
    m_specs: dict[int, list[MagneticSpec]],
) -> dict[str, np.ndarray]:
    base_log = log_value_fn(samples)
    return {
        "samples": samples,
        "A": star_local_values(log_value_fn, geom, samples, base_log),
        "B": triangle_local_values(geom, samples),
        "W_e": electric_local_values(samples, e_specs),
        "W_m": magnetic_local_values(log_value_fn, samples, base_log, m_specs),
    }
