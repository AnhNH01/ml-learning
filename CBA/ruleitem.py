class RuleItem:
    """
    Structute
    ----
    cond_set: set of item {name: value, name:value}

    class_label: class label

    cond_set_support_count: support count of condset--> the number of cases containing condset

    rule_support_count: support of rule --> the number of cases containing condset labeled class_label

    support = rule_support_count / (data_set size) * 100%

    confidence = rule_support_count / cond_set_support_count * 100%


    """

    # data_set is a list like [[1, 3, 1, 1, 'Iris-setosa'], [1, 1, 1, 1, 'Iris-setosa'],]

    def __init__(self, cond_set, class_label, data_set):
        self.cond_set = cond_set
        self.class_label = class_label
        (
            self.cond_set_support_count,
            self.rule_support_count,
        ) = self._get_support_count(data_set)
        self.confidence = self._get_confidence()
        self.support = self._get_support(len(data_set))

    def _get_support_count(self, data_set):
        """
        Count the number of cases containing cond_set in data_set, 
        if the containing case have the class name == class label, increment rule support
        """
        cond_set_support_count = 0
        rule_support_count = 0

        for case in data_set:
            for index in self.cond_set:
                contain = True
                if self.cond_set[index] != case[index]:
                    contain = False
                    break
            if contain:
                cond_set_support_count += 1
                class_name = case[-1]
                
                if self.class_label == class_name:
                    rule_support_count += 1

        return cond_set_support_count, rule_support_count


    def _get_support(self, data_set_size):
        return self.rule_support_count / data_set_size

    def _get_confidence(self):
        return (
            self.rule_support_count / self.cond_set_support_count
            if self.cond_set_support_count
            else 0
        )


if __name__ == '__main__':
    cond_set = {0: 1, 1: 1}
    class_label = 1
    dataset = [[1, 1, 1], [1, 1, 1], [1, 2, 1], [2, 2, 1], [2, 2, 1],
               [2, 2, 0], [2, 3, 0], [2, 3, 0], [1, 1, 0], [3, 2, 0]]
    rule_item = RuleItem(cond_set, class_label, dataset)
    
    print('condsupCount =', rule_item.cond_set_support_count)   # should be 3
    print('rulesupCount =', rule_item.rule_support_count)   # should be 2
    print('support =', rule_item.support)               # should be 0.2
    print('confidence =', rule_item.confidence)         # should be 0.667