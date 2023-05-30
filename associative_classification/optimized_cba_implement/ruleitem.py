from typing import Hashable, Iterable
from manager import DatasetManager


class RuleItem:
    def __init__(
        self,
        cond_set: set[tuple[Hashable, Hashable]],
        class_label: Hashable,
        manager: DatasetManager,
    ):
        self.cond_set = cond_set
        self.class_label = class_label
        (
            self.cond_set_support_count,
            self.rule_support_count,
        ) = manager.get_support_count(cond_set, class_label)
        self.transaction_num = manager.dataset_length

    @property
    def confidence(self):
        try:
            return self.rule_support_count / self.cond_set_support_count
        except Exception:
            return 0

    @property
    def support(self):
        try:
            return self.rule_support_count / self.transaction_num
        except Exception:
            return 0

    def __repr__(self) -> str:
        return f"{self.cond_set} --> (class: {self.class_label}) [support={self.support}, confidence={self.confidence}]"

    def __gt__(self, other):
        """
        Define precedent of rule according to the paper
        """
        if self.confidence > other.confidence:
            return True
        elif self.confidence == other.confidence:
            return self.support > other.support
        elif self.confidence == other.confidence and self.support == other.support:
            return len(self.cond_set) < len(other.cond_set)

    def __gt__(self, other):
        """
        precedence operator. Determines if this rule
        has higher precedence. Rules are sorted according
        to their confidence, support, length and id.
        """
        if self.confidence > other.confidence:
            return True
        elif self.confidence == other.confidence and self.support > other.support:
            return True
        elif (
            self.confidence == other.confidence
            and self.support == other.support
            and len(self.cond_set) < len(other.cond_set)
        ):
            return True

        else:
            return False

    def __lt__(self, other):
        return not self > other

    def __eq__(self, other: object) -> bool:
        if self.cond_set == other.cond_set and self.class_label == other.class_label:
            return True
        else:
            return False

    def __hash__(self) -> int:
        to_hash = set(self.cond_set)
        to_hash.add(("class", self.class_label))
        return hash(frozenset(to_hash))


if __name__ == "__main__":
    cond_set = {(0, 1), (1, 1)}
    class_label = 1
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

    manager = DatasetManager(dataset)

    rule_item = RuleItem(cond_set, class_label, manager)

    print(rule_item)
    print("condsupCount =", rule_item.cond_set_support_count)  # should be 3
    print("rulesupCount =", rule_item.rule_support_count)  # should be 2
    print("support =", rule_item.support)  # should be 0.2
    print("confidence =", rule_item.confidence)  # should be 0.667
