"""Parent selection and population management."""
from __future__ import annotations

import heapq
import random


def parent_selection(pop: list[dict], m: int) -> list[dict]:
    """Pick m parents weighted by rank (lower objective = higher weight)."""
    if not pop:
        raise ValueError("Cannot select parents from an empty population.")
    ranks = list(range(len(pop)))
    probs = [1 / (rank + 1 + len(pop)) for rank in ranks]
    return random.choices(pop, weights=probs, k=m)


def population_management(pop: list[dict], size: int) -> list[dict]:
    """Filter, dedup by objective, return top-`size` ascending."""
    pop = [ind for ind in pop if ind.get("objective") is not None]
    if not pop:
        return pop
    seen: set = set()
    unique: list[dict] = []
    for ind in pop:
        if ind["objective"] not in seen:
            seen.add(ind["objective"])
            unique.append(ind)
    return heapq.nsmallest(min(size, len(unique)), unique, key=lambda x: x["objective"])
