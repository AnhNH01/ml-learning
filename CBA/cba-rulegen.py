from ruleitem import RuleItem


class FrequentRuleItemsSet:
    """
    Frequent k-ruleitem set
    """

    def __init__(self) -> None:
        self.frequen_ruleitem_set = set()
    
    def add_ruleitem(self, rule_item):
        """
        Add new rule into frequent ruleset, only add new rule
        """
        for item in self.frequen_ruleitem_set:
            if item.class_label == rule_item.cond_set and item.cond_set == rule_item.cond_set:
                self.frequen_ruleitem_set.add(rule_item)
            else:
                break

    def append_ruleitem_sets(self, ruleitem_set):
        for item in ruleitem_set:
            self.add_ruleitem(item)

    

def get_frequent_one_rule_item(data_set, minsup):
    frequent_one_rule_item = set()

    class_lable = set(data[-1] for data in data_set)
    # for data in data_set:
    print(class_lable)
    for column in range(0, len(data_set[0])-1):
        values = set(data[column] for data in data_set)
        for value in values:
            condset = {column: value}
            for lable in class_lable:
                ruleitem = RuleItem(cond_set=condset, class_label=lable, data_set=data_set)
                if ruleitem.support > minsup:
                    frequent_one_rule_item.add(ruleitem)
    return frequent_one_rule_item




if __name__ == '__main__':
    dataset = [[1, 1, 1], [1, 1, 1], [1, 2, 1], [2, 2, 1], [2, 2, 1],
               [2, 2, 0], [2, 3, 0], [2, 3, 0], [1, 1, 0], [3, 2, 0]]
    minsup = 0.15
    minconf = 0.6


    frequent_one_rule_item = get_frequent_one_rule_item(dataset, minsup)
    print(frequent_one_rule_item)

    # frequent_one_rule_item = rule_generator(dataset, minconf, minsup)
    # print(rule_item for rule_item in frequent_one_rule_item)

