from read import read
from pre_processing import pre_process
from time import perf_counter
from cba_rg import rule_generator
from m1_algo import m1_classifier_builder
from manager import DatasetManager

def validate_with_dataset(train_data_path, test_data_path, data_scheme_path, minsup = 0.01, minconf=0.5):
    print(f"         CBA-implement at {minsup} support and {minconf} confidence ")
    print("----------------------------------------------------------------------------")
    data_train, attributes, value_type = read(train_data_path, data_scheme_path)
    data_test, attr, value_type = read(test_data_path,data_scheme_path)
    
    # dataset = pre_process(data, attributes, value_type)  
   
    manager = DatasetManager(data_train)
    
    # Generate rules
    start_time = perf_counter()
    cars = rule_generator(manager, minsup, minconf, max_length=len(data_train[0]) - 1)

    end_time = perf_counter()
    cba_rg_runtime = end_time - start_time
    
    # Bulding classifier
    start_time = perf_counter()
    classifier_m1 = m1_classifier_builder(cars.rules, data_train)
    end_time = perf_counter()
    cba_cb_runtime = end_time - start_time
    
    # Get accuracy score
    accuracy = classifier_m1.accuracy(data_test)
    
    print("Cars:", len(cars.rules))
    print(f"Default class: {classifier_m1.default_class}")
    print(f"Number of rules: {len(classifier_m1.rules)}")
    print(classifier_m1.rules[-1])
    print(f"CBA-RG runtime is {cba_rg_runtime}")
    print(f"CBA-CB runtime is {cba_cb_runtime}")
    print(f'Accuracy is {accuracy}')
    print("--------------------------------------------------------------------------")



if __name__ == "__main__":
   
    nursery_train = "/home/haianh/grad_project/ml-learning/benchmark-dataset/nursery-train.data"
    nursery_test = "/home/haianh/grad_project/ml-learning/benchmark-dataset/nursery-test.data"
    nursery_name = "/home/haianh/grad_project/ml-learning/benchmark-dataset/nursery.names"

    adult_train = "/home/haianh/grad_project/ml-learning/benchmark-dataset/adult-discretized-train.data"
    adult_test = "/home/haianh/grad_project/ml-learning/benchmark-dataset/adult-discretized-test.data"
    adult_name = "/home/haianh/grad_project/ml-learning/benchmark-dataset/adult.names"
    validate_with_dataset(adult_train,adult_test, adult_name, 0.03, 0.5)