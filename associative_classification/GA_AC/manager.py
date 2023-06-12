from typing import Iterable, Hashable
from collections import defaultdict


class DatasetManager:
    """
    Manager class for dataset

    Attributes:
        item is in the form of (attr, value)
        indices_by_item: hold indices of datacases containing item
        indices_by_class_label: hold indices of datacases containing class_label
        datacase_length: length of a datacase
    """

    def __init__(self, dataset: Iterable[Iterable[Hashable]]) -> None:
        self.indices_by_item: dict[tuple, set[int]] = defaultdict(set)
        self.dataset_length = len(dataset)
        self.indices_by_class_label: dict[Hashable, set[int]] = defaultdict(set)
        self.datacase_length =len(dataset[0]) 
        self.items = defaultdict(set)
        
        for index, datacase in enumerate(dataset):
            *data, class_label = datacase
            for attr, value in enumerate(data):
                self.indices_by_item[(attr, value)].add(index)
                self.items[attr].add(value)
            self.indices_by_class_label[class_label].add(index)

    def __len__(self):
        return self.dataset_length

    # @property
    # def items(self) -> list[tuple[Hashable, Hashable]]:
    #     """
    #     Might need to turn into dict of list to accomodate mutation
    #     """
    #     return list(self.indices_by_item.keys())

    @property
    def class_labels(self) -> list[Hashable]:
        return list(self.indices_by_class_label.keys())

    def indices_by_itemset(self, itemset: list[Hashable]) -> set[int]:
        """
        Return indices of datacases containing itemset
        by intersecting sets of indices of datacases containing item in itemset
        """

        filtered = {
            (attr, value) for (attr, value) in enumerate(itemset) if value is not None
        }
        if len(filtered) == 0: return filtered
        indices = self.indices_by_item[filtered.pop()]
        while filtered:
            item = filtered.pop()
            indices = self.indices_by_item[item].intersection(indices)

        return indices

    def get_support_count(
        self,
        cond_set: list[Hashable],
        class_label: Hashable,
    ) -> tuple[int, int]:
        """
        Return (cond_support_count and rule_support_count) for a ruleitem
        """
        indices_by_cond_set = self.indices_by_itemset(cond_set)
        cond_set_support_count = len(indices_by_cond_set)

        indices_by_class_label = self.indices_by_class_label[class_label]
        rule_support_count = len(
            indices_by_class_label.intersection(indices_by_cond_set)
        )

        return cond_set_support_count, rule_support_count


if __name__ == "__main__":
    dataset = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 2, 1],
        [2, 2, 1],
        [2, 2, 1],
        [2, 2, 0],
        [2, 3, 0],
        [2, 3, 0],
        [1, 1, 0],
        [3, 2, 0],
    ]
    minsup = 0.15
    minconf = 0.6

    manager = DatasetManager(dataset)
    print(manager.items)
    print(manager.class_labels)
    consupcount, rulesupcount = manager.get_support_count([2, 3], 0)
    print(consupcount, rulesupcount)
