"""Online Bin Packing task. Objective = mean (avg_bins_used - lb) / lb (lower is better)."""
from __future__ import annotations

from typing import Callable

import numpy as np

from eoh.tasks.base import BaseProblem
from eoh.tasks._obp_instances import GetData


class BPONLINE(BaseProblem):
    template_program = '''
def score(item: int, bins: np.ndarray) -> np.ndarray:
    """Score each bin for assigning the current item. Higher score = preferred bin.

    Args:
        item: size of the current item to assign
        bins: remaining capacities of feasible bins (all >= item size)
    Returns:
        scores: priority scores for each bin
    """
    return bins
'''
    task_description = (
        "Design a novel score function that scores a set of bins to assign an item. "
        "In each step, the item will be assigned to the bin with the maximum score. "
        "The final goal is to minimize the number of used bins."
    )

    def __init__(self, capacity: int = 100, timeout: int = 40, n_processes: int = 1):
        super().__init__(timeout=timeout, n_processes=n_processes)
        self.capacity = capacity
        self.instances, self.lb = GetData().get_instances(capacity)

    @staticmethod
    def get_valid_bin_indices(item: float, bins: np.ndarray) -> np.ndarray:
        return np.nonzero((bins - item) >= 0)[0]

    def online_binpack(self, items: tuple, bins: np.ndarray, score_func: Callable):
        packing = [[] for _ in bins]
        for item in items:
            valid = self.get_valid_bin_indices(item, bins)
            priorities = score_func(item, bins[valid])
            best = valid[np.argmax(priorities)]
            bins[best] -= item
            packing[best].append(item)
        return packing, bins

    def evaluate_program(self, program_str: str, callable_func: Callable) -> float | None:
        fitness_per_dataset = []
        for name, dataset in self.instances.items():
            num_bins_list = []
            for _, instance in dataset.items():
                capacity = instance["capacity"]
                items = np.array(instance["items"])
                bins = np.array([capacity] * instance["num_items"])
                _, bins_packed = self.online_binpack(items, bins, callable_func)
                num_bins_list.append(-(bins_packed != capacity).sum())
            avg = -np.mean(num_bins_list)
            fitness_per_dataset.append((avg - self.lb[name]) / self.lb[name])
        return float(np.mean(fitness_per_dataset))
