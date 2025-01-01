from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Generator, Tuple

import tqdm
from rapidfuzz import distance

__all__ = ["BKTree"]


def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) > len(s2):
        # get the shorter string (may be cached)
        return levenshtein_distance(s2, s1)
    return distance.DamerauLevenshtein.distance(s1, s2)


class BKTreeNode:
    def __init__(self, text: str):
        self.children: Dict[int, BKTreeNode] = {}
        self.text = text

    def add(self, text: str):
        node = self
        while True:
            distance = levenshtein_distance(text, node.text)
            if distance == 0:
                # word already exists
                return
            next_node = node.children.get(distance)
            if not next_node:
                node.children[distance] = BKTreeNode(text)
                break
            node = next_node

    def search(
        self, text: str, threshold: float = 2
    ) -> Generator[Tuple[str, int], None, None]:
        distance = levenshtein_distance(text, self.text)
        if distance <= threshold:
            yield (self.text, distance)
        # Search children within distance range
        for d in range(distance - threshold, distance + threshold + 1):
            child = self.children.get(d)
            if child:
                yield from child.search(text, threshold)


class BKTree:
    def __init__(self):
        self.root = None

    def add(self, text: str):
        if not self.root:
            self.root = BKTreeNode(text)
        else:
            self.root.add(text)

    def search(self, text: str, threshold: float):
        if not self.root:
            return []
        results = self.root.search(text, threshold)
        return sorted(results, key=lambda x: x[1])

    def dump(self, path: str | Path):
        self.root.dump(path)

    @classmethod
    def load(cls, path: str | Path) -> BKTree:
        tree = cls.__new__(cls)
        tree.root = BKTreeNode.load(path)
        return tree


def build_bktree(keywords: list[str], seed: int = 0) -> BKTree:
    random.seed(seed)
    # might speed up the process by shuffling the keywords
    keywords = random.sample(keywords, len(keywords))
    # Helper function to build the tree from a list
    tree = BKTree()
    for keyword in tqdm.tqdm(keywords):
        # For phrases, add both the full phrase and individual words
        tree.add(keyword.lower())
    return tree
