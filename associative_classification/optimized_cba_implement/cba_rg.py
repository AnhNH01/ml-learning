from ruleitem import RuleItem
from manager import DatasetManager
from itertools import combinations


class CAR:
    """
    Set of class asscociation rule, satisfying minimum support and minimum confidece
    """

    def __init__(self) -> None:
        self.rules: set[RuleItem] = set()
        self.pruned_rules: set[RuleItem] = set()

    def _add(self, rule: RuleItem, min_support: float, min_confidence: float):
        """
        Save the rule with the highest confidence if they have the same cond_set
        """
       
        if rule.support >= min_support and rule.confidence >= min_confidence:
            if rule in self.rules:
                return
            for existing_rule in self.rules:
                if existing_rule.cond_set == rule.cond_set and existing_rule.confidence < rule.confidence:
                    self.rules.remove(existing_rule)
                    self.rules.add(rule)
                    return
                elif existing_rule.cond_set == rule.cond_set and existing_rule.confidence >= rule.confidence:
                    return
            self.rules.add(rule)

    def append(self, car, min_support: float, min_confidence: float):
        """
        Append car
        """
        for rule in car.rules:
            self._add(rule, min_support, min_confidence)

    def generate_car(
        self,
        frequent_ruleitem: set[RuleItem],
        min_support: float,
        min_confidence: float,
    ):
        """
        Get rules that satisfied min support and min_confidence
        """
        for rule in frequent_ruleitem:
            self._add(rule, min_support, min_confidence)


def apriori_gen(k_ruleitem_set: set[RuleItem], manager: DatasetManager):
    """
    Use the apriori_gen procedure to generate k+1 cond_set length rule set

    Params:

    k_ruleitem_set = k-length condset ruleitem set

    manager: manager used for creating rules
    """

    candidate_ruleitem: set[RuleItem] = set()

    rule_list = list(k_ruleitem_set)

    candidate_cond_set = set()
    # Join step
    i = 0
    while i < len(rule_list):
        *first, last = sorted(rule_list[i].cond_set)
        label = rule_list[i].class_label
        tail = [last]
        for j in range(0, len(rule_list)):
            if rule_list[j] == rule_list[i]:
                continue
            if rule_list[j].class_label != label:
                continue
            else:
                *first_items, last_item = sorted(rule_list[j].cond_set)

                if first == first_items:
                    tail.append(last_item)
                else:
                    break
                

        for combination in combinations(tail, 2):
            extended = frozenset(sorted(first + list(combination)))
            candidate_cond_set.add(extended)
            

        for condset in candidate_cond_set:
            candidate = RuleItem(set(condset), label, manager)
            candidate_ruleitem.add(candidate)
            
        i += 1 # increment while-loop
    
    # Prune step
    # pruned_candidates = set()
    
    # for rule in candidate_ruleitem:
    #     for i in range(0, len(rule.cond_set)):
    #         condset = list(rule.cond_set)
    #         cond_subset = set(condset[:i] + condset[i + 1 :])
    #         temp = RuleItem(cond_subset, rule.class_label, manager)

    #         if temp not in k_ruleitem_set:                
    #             # print("pruned a rule")
    #             break
    #         else:
    #             pruned_candidates.add(rule)
    return candidate_ruleitem


def rule_generator(
    manager: DatasetManager,
    min_support: float,
    min_confidence: float,
    max_length: int,
):
    """
    Buid up ruleitem from the 1-ruleitem set
    """

    cars = CAR()
    frequent_ruleitem_set: dict[int, set[RuleItem]] = dict()
   
    # generate 1-ruleitem set
    cond_set_length = 1
    one_ruleiem_set = set()
    for label in manager.class_labels:
        for item in manager.items:
            cond_set = set()
            cond_set.add(item)
            rule = RuleItem(cond_set, label, manager)
            if rule.support >= min_support:
                one_ruleiem_set.add(rule)
    frequent_ruleitem_set[cond_set_length] = one_ruleiem_set
    cars.generate_car(one_ruleiem_set, min_support, min_confidence)
    
    # build up k+1-ruleiem set
    cond_set_length = 2
    while len(frequent_ruleitem_set[cond_set_length - 1]) > 0 and max_length > 1:
       
        rule_list = list(frequent_ruleitem_set[cond_set_length - 1])

        candidate_rules = apriori_gen(rule_list, manager)
        car = CAR()
        
        large_possible_rules = set()
        for rule in candidate_rules:
            if rule.support >= min_support:
                large_possible_rules.add(rule)
        if len(large_possible_rules) == 0:
            break
        frequent_ruleitem_set[cond_set_length] = large_possible_rules
        car.generate_car(large_possible_rules, min_support, min_confidence)
        
        cars.append(car, min_support, min_confidence)
        cond_set_length += 1

        if cond_set_length >= max_length:
            break


    return cars

if __name__ == "__main__":
    from time import perf_counter
    dataset = [[1, 1, 1], [1, 1, 1], [1, 2, 1], [2, 2, 1], [2, 2, 1],
               [2, 2, 0], [2, 3, 0], [2, 3, 0], [1, 1, 0], [3, 2, 0]]
    minsup = 0.15

    minconf = 0.6


    def runtest(datapath, namepath, minsup, minconf):
        from read import read
        from pre_processing import pre_process
        from time import perf_counter

        data, attr, value_type = read(datapath, namepath)
        # result_data = pre_process(data, attr, value_type)
        manager = DatasetManager(data)
        s = perf_counter()
        cars = rule_generator(manager, minsup, minconf, len(data[0]) - 1)
        e = perf_counter()
        
        for rule in sorted(cars.rules, reverse=True):
            print(rule)
        
        print(f"CBA-RG done in {e-s} second")
        print(f"Number of rules: {len(cars.rules)}")
       

    # manager = DatasetManager(dataset)
    # start = perf_counter()
    # cars = rule_generator(manager, minsup, minconf, 2)
    # end = perf_counter()
   
    
    # print(f"Done in {end-start} secs")
    # for rule in cars.rules:
    #     print(rule)

    # datapath = "/home/haianh/grad_project/ml-learning/data/adult.data"
    # namepath = "/home/haianh/grad_project/ml-learning/data/adult.names"
    # runtest(datapath, namepath, 0.1, 0.5)

    dress_data = "/home/haianh/grad_project/CBA/datasets/dresses.data"
    dresses_name = "/home/haianh/grad_project/CBA/datasets/dresses.names"
    runtest(dress_data, dresses_name, 0.03, 0.5)