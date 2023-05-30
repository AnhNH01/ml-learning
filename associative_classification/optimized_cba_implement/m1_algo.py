from classifier import Classifier, satisfy
from copy import deepcopy



def m1_classifier_builder(car, dataset:list):
    classifier = Classifier()
    car_list = sorted(car, reverse=True) # sort class association rules by its precedent
    rule_error_list = []

    for rule in car_list:
        to_remove = []
        mark = False

        cases_covered = 0
        cases_correct = 0

        # check if rule correctly classify at least one datacase
        for index, datacase in enumerate(dataset):
            flag = satisfy(rule, datacase)
            if flag is not None: # the datacase is covered
                cases_covered += 1
                to_remove.append(index) # add the index of datacase to be removed
                if flag == True: # correctly classified the datatcase
                    mark = True
                    cases_correct += 1
        # if there are at least 1 datacase correctly classified, remove all covered datacases
        if mark:
            tmp_dataset = deepcopy(dataset)
            for index in to_remove:
                tmp_dataset[index] = []
            while [] in tmp_dataset:
                tmp_dataset.remove([])

            dataset = tmp_dataset

            # insert rule, calculate error rate, select default class based on the dataset
            classifier.insert_rule(rule)
            classifier.select_default_class(dataset)        
            #classifier.compute_error(dataset)

            # compute total error = rule error + default class error

            # rule error
            rule_error_list.append(cases_covered - cases_correct)

            # default class error
            class_col = [datacase[-1] for datacase in dataset]
            default_class_error = len(class_col) - class_col.count(classifier._default_class_list[-1])
            
            # total error list
            classifier._error_list.append(sum(rule_error_list) + default_class_error)


    classifier.discard()
    return classifier




if __name__ == '__main__':
    from cba_rg import rule_generator
    from manager import DatasetManager
    from read import read

    # dataset = [[1, 1, 1], [1, 1, 1], [1, 2, 1], [2, 2, 1], [2, 2, 1],
    #            [2, 2, 0], [2, 3, 0], [2, 3, 0], [1, 1, 0], [3, 2, 0]]
    
    
    # minsup = 0.15
    # minconf = 0.6
    
    # manager = DatasetManager(dataset)

    # cars = rule_generator(manager, minsup, minconf, max_length=len(dataset[0]) - 1)
    # classifier = m1_classifier_builder(cars.rules, dataset)
    # classifier.print()

    # print(classifier.accuracy(dataset))
   
    dress_data = "/home/haianh/grad_project/CBA/datasets/dresses.data"
    dresses_name = "/home/haianh/grad_project/CBA/datasets/dresses.names"

    data, attr, value_type = read(dress_data, dresses_name)
        # result_data = pre_process(data, attr, value_type)
    manager = DatasetManager(data)
    cars = rule_generator(manager, 0.01, 0.5, len(data[0]) - 1)

    classifier = m1_classifier_builder(cars.rules, data)
    # classifier.print()
    print(len(classifier.rules))
    print(classifier.accuracy(data))