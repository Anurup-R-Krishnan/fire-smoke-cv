from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(image: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    """Return a 64-bit DCT perceptual hash.

    The implementation is local so Data Prep does not depend on a separate image-hash
    package. Similar images generally have hashes with a small Hamming distance.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot hash an empty image")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    size = hash_size * highfreq_factor
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))
    low = dct[:hash_size, :hash_size]
    # Ignore the DC coefficient when choosing the threshold.
    median = float(np.median(low.flatten()[1:]))
    bits = low > median
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bool(bit))
    return value


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


class BKTree:
    """BK-tree for efficient Hamming-distance lookup of 64-bit hashes."""

    def __init__(self) -> None:
        self._root: tuple[int, dict[int, object]] | None = None

    def add(self, value: int) -> None:
        if self._root is None:
            self._root = (value, {})
            return
        node_value, children = self._root
        while True:
            distance = hamming_distance(value, node_value)
            child = children.get(distance)
            if child is None:
                children[distance] = (value, {})
                return
            node_value, children = child  # type: ignore[assignment]

    def query(self, value: int, max_distance: int) -> list[int]:
        if self._root is None:
            return []
        results: list[int] = []
        stack = [self._root]
        while stack:
            node_value, children = stack.pop()
            distance = hamming_distance(value, node_value)
            if distance <= max_distance:
                results.append(node_value)
            lower = distance - max_distance
            upper = distance + max_distance
            for edge, child in children.items():
                if lower <= edge <= upper:
                    stack.append(child)  # type: ignore[arg-type]
        return results


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1
