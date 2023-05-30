from read import read
from pre_processing import pre_process
import time
from cba_rg import rule_generator
from m1_algo import m1_classifier_builder
from manager import DatasetManager
from sklearn.model_selection import train_test_split

def validate_with_dataset(data_path, data_scheme_path, test_split = 0.2, minsup = 0.01, minconf=0.5):
    data, attributes, value_type = read(data_path, data_scheme_path)
    
    # dataset = pre_process(data, attributes, value_type)  
    dataset = data
    data_train, data_test = train_test_split(dataset, shuffle=True, test_size=test_split)
    # data_train = dataset
    manager = DatasetManager(data_train)
  
    
    start_time = time.time()
    cars = rule_generator(manager, minsup, minconf, max_length=len(dataset[0]) - 1)
    # for car in sorted(cars.rules, reverse=True):
    #     print(car)
    end_time = time.time()
    cba_rg_runtime = end_time - start_time
    
    start_time = time.time()
    classifier_m1 = m1_classifier_builder(cars.rules, data_train)
    end_time = time.time()
    cba_cb_runtime = end_time - start_time
    accuracy = classifier_m1.accuracy(data_test)
    print("Cars:", len(cars.rules))
    print(f"Default class: {classifier_m1.default_class}")
    print(f"Number of rules: {len(classifier_m1.rules)}")
    # print(classifier_m1.rules)
    print(f"CBA-RG runtime is {cba_rg_runtime}")
    print(f"CBA-CB runtime is {cba_cb_runtime}")
    print(f'Accuracy is {accuracy}')


if __name__ == "__main__":
    dress_data = "/home/haianh/grad_project/CBA/datasets/dresses.data"
    dresses_name = "/home/haianh/grad_project/CBA/datasets/dresses.names"

    adult_data = "/home/haianh/grad_project/ml-learning/data/adult.data"
    adult_name = "/home/haianh/grad_project/ml-learning/data/adult.names"

    iris_data = "/home/haianh/grad_project/ml-learning/data/iris.data"
    iris_name = "/home/haianh/grad_project/ml-learning/data/iris.names"

    nursery_data = "/home/haianh/grad_project/ml-learning/data/nursery.data"
    nursery_name = "/home/haianh/grad_project/ml-learning/data/nursery.names"
    validate_with_dataset(nursery_data, nursery_name, 0.2, 0.05, 0.5)