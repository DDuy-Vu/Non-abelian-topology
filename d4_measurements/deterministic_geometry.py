"""Deterministic D4 geometry and closed-form measurement operator pieces."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


COLOR_NAMES = {0: "red", 1: "green", 2: "blue"}


def coor2in(x: int, y: int, L: int) -> int:
    return ((y + L) % L) * L + ((x + L) % L)


@dataclass(frozen=True)
class Geometry:
    L: int
    N_plaquette: int
    N: int
    plaquette_list: np.ndarray
    x_list: np.ndarray
    left_triangles: np.ndarray
    right_triangles: np.ndarray
    plaquette_color_labels: np.ndarray
    site_color_labels: np.ndarray
    transform_matrix: np.ndarray

    @classmethod
    def build(cls, L: int) -> "Geometry":
        N_plaquette = L * L
        N = 3 * L * L
        plaquette_list = np.array(
            [
                [
                    3 * coor2in(i, j, L),
                    3 * coor2in(i + 1, j, L) + 2,
                    3 * coor2in(i + 1, j, L) + 1,
                    3 * coor2in(i + 1, j + 1, L),
                    3 * coor2in(i + 1, j + 1, L) + 2,
                    3 * coor2in(i, j, L) + 1,
                ]
                for j in range(L)
                for i in range(L)
            ],
            dtype=np.int64,
        )
        left_triangles = np.array(
            [
                [
                    3 * coor2in(i, j, L),
                    3 * coor2in(i + 1, j, L) + 1,
                    3 * coor2in(i + 1, j + 1, L) + 2,
                ]
                for j in range(L)
                for i in range(L)
            ],
            dtype=np.int64,
        )
        right_triangles = np.array(
            [
                [
                    3 * coor2in(i + 1, j, L) + 2,
                    3 * coor2in(i + 1, j + 1, L),
                    3 * coor2in(i, j, L) + 1,
                ]
                for j in range(L)
                for i in range(L)
            ],
            dtype=np.int64,
        )
        x_list = np.array(
            [
                [
                    3 * coor2in(i, j, L) + 2,
                    3 * coor2in(i, j - 1, L) + 1,
                    3 * coor2in(i + 1, j, L),
                    3 * coor2in(i + 2, j + 1, L) + 2,
                    3 * coor2in(i + 1, j + 1, L) + 1,
                    3 * coor2in(i, j + 1, L),
                ]
                for j in range(L)
                for i in range(L)
            ],
            dtype=np.int64,
        )

        transform_matrix = np.zeros((N, N_plaquette), dtype=np.int8)
        for p in range(N_plaquette):
            transform_matrix[x_list[p], p] = 1

        plaquette_color_labels = np.full(N_plaquette, -1, dtype=np.int8)
        site_color_labels = np.full(N, -1, dtype=np.int8)
        if L % 3 != 0:
            raise ValueError("Color-resolved D4 measurements require L divisible by 3.")
        for y in range(L):
            for x in range(L):
                p = coor2in(x, y, L)
                plaquette_color_labels[p] = (x + y) % 3
        for color in (0, 1, 2):
            plaquettes = np.flatnonzero(plaquette_color_labels == color)
            sites = np.unique(x_list[plaquettes].reshape(-1))
            site_color_labels[sites] = color

        return cls(
            L=L,
            N_plaquette=N_plaquette,
            N=N,
            plaquette_list=plaquette_list,
            x_list=x_list,
            left_triangles=left_triangles,
            right_triangles=right_triangles,
            plaquette_color_labels=plaquette_color_labels,
            site_color_labels=site_color_labels,
            transform_matrix=transform_matrix,
        )

    def triangles(self) -> list[tuple[str, int, np.ndarray]]:
        out: list[tuple[str, int, np.ndarray]] = []
        for p, tri in enumerate(self.left_triangles):
            out.append(("L", p, tri))
        for p, tri in enumerate(self.right_triangles):
            out.append(("R", p, tri))
        return out


@dataclass(frozen=True)
class Spec:
    flips: tuple[int, ...]
    cz_pairs: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class PaperMagneticString:
    spec: Spec
    path_nodes: tuple[tuple[str, int, int], ...]
    passed_sites: tuple[int, ...]


def parity_tuple(items):
    counts = {}
    for item in items:
        counts[item] = 1 - counts.get(item, 0)
    return tuple(sorted(item for item, bit in counts.items() if bit))


def normalize_pairs(pairs):
    return parity_tuple(tuple(sorted((int(a), int(b)))) for a, b in pairs)


def star_ring_pairs(geom: Geometry, p: int) -> tuple[tuple[int, int], ...]:
    inner = geom.plaquette_list[p]
    return normalize_pairs((inner[a], inner[(a + 1) % 6]) for a in range(6))


def all_to_all_pairs_for_sequence(
    geom: Geometry, color: int, sequence: list[int]
) -> tuple[tuple[int, int], ...]:
    left_color = (color - 1) % 3
    right_color = (color + 1) % 3
    pairs = []
    for i, site_i in enumerate(sequence):
        if int(geom.site_color_labels[site_i]) != left_color:
            continue
        for site_j in sequence[i + 1 :]:
            if int(geom.site_color_labels[site_j]) == right_color:
                pairs.append((int(site_i), int(site_j)))
    return normalize_pairs(pairs)


def cz_value(samples: np.ndarray, i: int, j: int) -> np.ndarray:
    return (1 + samples[:, i] + samples[:, j] - samples[:, i] * samples[:, j]) // 2


def phase(samples: np.ndarray, pairs: tuple[tuple[int, int], ...]) -> np.ndarray:
    out = np.ones(samples.shape[0], dtype=np.int8)
    for i, j in pairs:
        out *= cz_value(samples, i, j).astype(np.int8)
    return out


def flip_samples(samples: np.ndarray, flips: tuple[int, ...]) -> np.ndarray:
    out = samples.copy()
    if flips:
        out[:, list(flips)] *= -1
    return out


def triangle_anticommutators(geom: Geometry, spec: Spec) -> list[tuple[str, int, int]]:
    flips = set(spec.flips)
    out = []
    for orient, p, tri in geom.triangles():
        overlap = sum(1 for site in tri if int(site) in flips)
        if overlap % 2:
            color = int(geom.site_color_labels[int(tri[0])])
            out.append((orient, p, color))
    return out


def honeycomb_node_color(node: tuple[str, int, int]) -> int:
    orient, x, y = node
    if orient == "L":
        return (x + y - 1) % 3
    return (x + y + 1) % 3


def site_index_from_lift(geom: Geometry, x: int, y: int, sublattice: int) -> int:
    return 3 * coor2in(x, y, geom.L) + sublattice


def honeycomb_neighbors(
    geom: Geometry, node: tuple[str, int, int]
) -> list[tuple[tuple[str, int, int], int]]:
    orient, x, y = node
    if orient == "L":
        return [
            (("R", x - 1, y - 1), site_index_from_lift(geom, x, y, 0)),
            (("R", x + 1, y), site_index_from_lift(geom, x + 1, y, 1)),
            (("R", x, y + 1), site_index_from_lift(geom, x + 1, y + 1, 2)),
        ]
    return [
        (("L", x + 1, y + 1), site_index_from_lift(geom, x + 1, y + 1, 0)),
        (("L", x - 1, y), site_index_from_lift(geom, x, y, 1)),
        (("L", x, y - 1), site_index_from_lift(geom, x + 1, y, 2)),
    ]


def quotient_node(geom: Geometry, node: tuple[str, int, int]) -> tuple[str, int, int]:
    return node[0], node[1] % geom.L, node[2] % geom.L


def same_torus_node_path(
    geom: Geometry, color: int, start: tuple[str, int, int], end: tuple[str, int, int]
) -> tuple[tuple[tuple[str, int, int], ...], tuple[int, ...]]:
    start_q = quotient_node(geom, start)
    end_q = quotient_node(geom, end)
    queue = deque([start_q])
    parent: dict[tuple[str, int, int], tuple[tuple[str, int, int], int] | None] = {
        start_q: None
    }
    while queue:
        node = queue.popleft()
        if node == end_q:
            break
        for nxt, site in honeycomb_neighbors(geom, node):
            if honeycomb_node_color(nxt) != color:
                raise AssertionError(("bad honeycomb color", color, node, nxt))
            nxt_q = quotient_node(geom, nxt)
            if nxt_q in parent:
                continue
            parent[nxt_q] = (node, site)
            queue.append(nxt_q)

    if end_q not in parent:
        raise ValueError(("no open path", color, start_q, end_q))

    nodes = []
    sites = []
    current = end_q
    while current != start_q:
        prev_entry = parent[current]
        if prev_entry is None:
            raise AssertionError("unexpected empty parent")
        prev, site = prev_entry
        nodes.append(current)
        sites.append(site)
        current = prev
    nodes.append(start_q)
    return tuple(reversed(nodes)), tuple(reversed(sites))


def same_torus_node_path_flips(
    geom: Geometry, color: int, start: tuple[str, int, int], end: tuple[str, int, int]
) -> tuple[int, ...]:
    _, flips = same_torus_node_path(geom, color, start, end)
    return flips


def inner_ring_for_honeycomb_node(
    geom: Geometry, node: tuple[str, int, int]
) -> tuple[int, ...]:
    _, x, y = node
    return tuple(int(site) for site in geom.plaquette_list[coor2in(x, y, geom.L)])


def passed_site_between_flips(
    geom: Geometry, node: tuple[str, int, int], flip_a: int, flip_b: int
) -> int:
    ring = inner_ring_for_honeycomb_node(geom, node)
    idx_a = ring.index(int(flip_a))
    idx_b = ring.index(int(flip_b))
    if (idx_a + 2) % 6 == idx_b:
        return int(ring[(idx_a + 1) % 6])
    if (idx_b + 2) % 6 == idx_a:
        return int(ring[(idx_b + 1) % 6])
    raise ValueError(("flips do not meet at a color triangle", node, flip_a, flip_b, ring))


def paper_magnetic_string(
    geom: Geometry, color: int, start: tuple[str, int, int], end: tuple[str, int, int]
) -> PaperMagneticString:
    """Construct the ordered line dressing of the paper's magnetic string.

    The measured opposite-orientation strings are represented by paths on the
    color-c triangle honeycomb graph.  The paper orientation is fixed here as
    traversal from the R triangle endpoint to the L triangle endpoint.  Along
    that oriented path, every internal color-c triangle contributes the
    non-color-c site passed between two consecutive flipped color-c sites.
    The CZ dressing is the ordered all-to-all product between the two other
    color sublattices, with the left/right color convention used by
    all_to_all_pairs_for_sequence.
    """

    nodes, flips = same_torus_node_path(geom, color, start, end)
    if nodes[0][0] == "L" and nodes[-1][0] == "R":
        nodes = tuple(reversed(nodes))
        flips = tuple(reversed(flips))
    elif not (nodes[0][0] == "R" and nodes[-1][0] == "L"):
        raise ValueError(("paper magnetic strings require opposite endpoints", color, start, end))

    passed_sites = tuple(
        passed_site_between_flips(geom, nodes[i + 1], flips[i], flips[i + 1])
        for i in range(len(flips) - 1)
    )
    cz_pairs = all_to_all_pairs_for_sequence(geom, color, list(passed_sites))
    return PaperMagneticString(
        spec=Spec(flips=tuple(int(site) for site in flips), cz_pairs=cz_pairs),
        path_nodes=tuple(nodes),
        passed_sites=passed_sites,
    )


def representative_open_paths(
    geom: Geometry, color: int, count: int
) -> list[tuple[tuple[str, int, int], tuple[str, int, int], tuple[int, ...]]]:
    nodes = [
        (orient, x, y)
        for y in range(geom.L)
        for x in range(geom.L)
        for orient in ("L", "R")
        if honeycomb_node_color((orient, x, y)) == color
    ]
    start = nodes[0]
    candidates = []
    for end in nodes[1:]:
        if end[0] == start[0]:
            continue
        flips = same_torus_node_path_flips(geom, color, start, end)
        endpoints = triangle_anticommutators(geom, Spec(flips=flips, cz_pairs=()))
        if len(endpoints) != 2:
            continue
        candidates.append((start, end, flips))
    candidates.sort(key=lambda item: (-len(item[2]), item[1]))

    out = []
    seen_lengths = set()
    for item in candidates:
        length = len(item[2])
        if length in seen_lengths and len(out) < count - 1:
            continue
        out.append(item)
        seen_lengths.add(length)
        if len(out) == count:
            return out
    for item in candidates:
        if item in out:
            continue
        out.append(item)
        if len(out) == count:
            return out
    raise ValueError(("not enough representative open paths", color, len(out), count))
