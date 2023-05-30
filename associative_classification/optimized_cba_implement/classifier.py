from ruleitem import RuleItem
import sys

def satisfy(rule: RuleItem, datacase):
    """
    Check if a rule cover and if it satisfy the datacase.

    The rule cover the datacase if the datacase matches the cond_set of the rule

    the rule satisfy the datacase if class_label of the rule matches the class label of the datacase
    """

    flag = False
    for (attr, value) in rule.cond_set:
        if datacase[attr] != value:
            return None # the rule doesn't cover the datacase
    else: # the rule covers the datacase
        if rule.class_label == datacase[-1]:
            flag = True
    
    return flag

class Classifier:
    def __init__(self) -> None:
        self.rules: list[RuleItem] = list()
        self.default_class = None
        self._error_list = list()
        self._default_class_list = list()

    def insert_rule(self, rule: RuleItem):
        """
        Insert the rule in to the classifier
        """
        self.rules.append(rule)
        
        pass

    def select_default_class(self, dataset):
        """
        Select the default class (majority class) in the remaining data
        """

        class_column = [x[-1] for x in dataset]
        class_label = set(class_column)
        max = 0
        current_default_class = None
        for label in class_label:
            if class_column.count(label) >= max:
                max = class_column.count(label)
                current_default_class = label
        self._default_class_list.append(current_default_class)

    def compute_error(self, dataset):
        if len(dataset) <= 0:
            self._error_list.append(sys.maxsize)
            return

        error_number = 0

        # the number of errors that have been made by all the selected rules in C
        for case in dataset:
            is_cover = False
            for rule in self.rules:
                if satisfy(rule, case):
                    is_cover = True
                    break
            if not is_cover:
                error_number += 1
        
        # the number of errors to be made by the default class in the training set
        class_column = [x[-1] for x in dataset]
        if self._default_class_list:
            error_number += len(class_column) - class_column.count(self._default_class_list[-1])
        self._error_list.append(error_number)

    def discard(self):
        # find the first rule p in C with the lowest total number of errors and drop all the rules after p in C
        index = self._error_list.index(min(self._error_list))
        self.rules = self.rules[:(index+1)]
        self._error_list = None

        # assign the default class associated with p to default_class
        self.default_class = self._default_class_list[index]
        self._default_class_list = None

    def print(self):
        print("_______Classifier_________")
        print("Rules:")
        for rule in self.rules:
            print(rule)

        print(f"Default class: {self.default_class}")
    
    def classify(self, datacase):
        """
        Classify datacase based on rules in the classifier

        Return True if correctly classified, False otherwise
        """
    
        for rule in self.rules:
            # return the class label of the first rule that covers the datacase
            if satisfy(rule, datacase) is not None:
                return rule.class_label
        else:
            return self.default_class
    
    def accuracy(self, dataset): 
        if len(self.rules) == 0 or self.default_class is None:
            print("The classifier has to be built first")
            return
        else:
            error_count = 0
            for datacase in dataset:
                label = datacase[-1]
                prediction = self.classify(datacase)
                if prediction != label:
                    error_count += 1
            
            return (1 - error_count / len(dataset)) * 100