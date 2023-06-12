from typing import Hashable
from manager import DatasetManager

# To consider: add rule error rate into the fitness function


class RuleItem:
    """
    Class representing RuleItem

    Attributes:
        cond_set: list of value, attribute is the index
        class_label: label of the class
        manager: DatasetManager instance for counting support
        manager: manager instance
    """
    def __init__(
        self,
        cond_set: list[Hashable],
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
        self.manager = manager

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

    @property
    def fitness(self):
        return self.fitness_function()
    
    @property
    def cond_set_length(self):
        """Length of the condset = Number of not None value in condset"""
        return sum(val is not None for val in self.cond_set)


    def fitness_function(self):
        """
        Fitness function for rule, takes into account support of the rule and
        perspective to support of the class

        F(rule) = confidence * support  / class_support

        Return:
            Fitness of a rule: float value in range [0, 1], the higher the better
        """
        class_support = len(self.manager.indices_by_class_label[self.class_label]) / self.transaction_num
        fitness = self.confidence * self.support / class_support
        return fitness

    def __repr__(self) -> str:
        filtered = [(attr, val) for (attr, val) in enumerate(self.cond_set) if val is not None]
        return f"{tuple(filtered)} --> (class: {self.class_label}) [sup={self.support}, conf={self.confidence}, fit = {self.fitness}]"

    def __gt__(self, other):
        """
        Precedence operator. Determines if this rule
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
        if tuple(self.cond_set) == tuple(other.cond_set) and self.class_label == other.class_label:
            return True
        else:
            return False

    def __hash__(self) -> int:
        to_hash = tuple(enumerate(self.cond_set)) + ("class", self.class_label)
        return hash(to_hash)

    def __len__(self) -> int:
        """
        Define "real" length of the rule: The number of not None value in the cond_set and ante
        """
        return  self.cond_set_length+ 1

if __name__ == "__main__":
    cond_set = [1, 1] 
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
    print('rule length =', len(rule_item)) # should be 3