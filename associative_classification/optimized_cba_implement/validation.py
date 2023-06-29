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
    
    print(f"CBA-RG runtime is {cba_rg_runtime}")
    print(f"CBA-CB runtime is {cba_cb_runtime}")
    print(f'Accuracy is {accuracy}')

    # classifier_m1.print()
    print("--------------------------------------------------------------------------")



if __name__ == "__main__":
   
    nursery_train = "/home/haianh/grad_project/ml-learning/benchmark-dataset/nursery-train.data"
    nursery_test = "/home/haianh/grad_project/ml-learning/benchmark-dataset/nursery-test.data"
    nursery_name = "/home/haianh/grad_project/ml-learning/benchmark-dataset/nursery.names"

    adult_train = "/home/haianh/grad_project/ml-learning/benchmark-dataset/adult-discretized-train.data"
    adult_test = "/home/haianh/grad_project/ml-learning/benchmark-dataset/adult-discretized-test.data"
    adult_name = "/home/haianh/grad_project/ml-learning/benchmark-dataset/adult.names"

    iris_train = "/home/haianh/grad_project/ml-learning/benchmark-dataset/iris-train.data"
    iris_test = "/home/haianh/grad_project/ml-learning/benchmark-dataset/iris-test.data"
    iris_name = "/home/haianh/grad_project/ml-learning/benchmark-dataset/iris.names"

    letter_train = "/home/haianh/grad_project/ml-learning/benchmark-dataset/letter-train.data"
    letter_test = "/home/haianh/grad_project/ml-learning/benchmark-dataset/letter-test.data"
    letter_name = "/home/haianh/grad_project/ml-learning/benchmark-dataset/letter.names"

    letterd_train = "/home/haianh/grad_project/ml-learning/benchmark-dataset/letter-discretized-train.data"
    letterd_test = "/home/haianh/grad_project/ml-learning/benchmark-dataset/letter-discretized-test.data"

    bank_train = "/home/haianh/grad_project/ml-learning/benchmark-dataset/bank-train.data"
    bank_test = "/home/haianh/grad_project/ml-learning/benchmark-dataset/bank-test.data"
    bank_name = "/home/haianh/grad_project/ml-learning/benchmark-dataset/bank.names"

    krkopt_train = "/home/haianh/grad_project/ml-learning/benchmark-dataset/krkopt-train.data"
    krkopt_test = "/home/haianh/grad_project/ml-learning/benchmark-dataset/krkopt-test.data"
    krkopt_name = "/home/haianh/grad_project/ml-learning/benchmark-dataset/krkopt.names"

    mushroom_train = "/home/haianh/grad_project/ml-learning/benchmark-dataset/mushroom-train.data"
    mushroom_test = "/home/haianh/grad_project/ml-learning/benchmark-dataset/mushroom-test.data"
    mushroom_name = "/home/haianh/grad_project/ml-learning/benchmark-dataset/mushroom.names"


    validate_with_dataset(adult_train, adult_test, adult_name, 0.1, 0.5)