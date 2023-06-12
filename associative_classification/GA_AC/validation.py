from read import read
from pre_processing import pre_process
from time import perf_counter
from ga_rg import ga_genrules
from m1_algo import m1_classifier_builder
from manager import DatasetManager


def validate_with_dataset(
    train_data_path, test_data_path, data_scheme_path, minsup=0.01, minconf=0.5
):
    print(f"         CBA-implement at {minsup} support and {minconf} confidence ")
    print(
        "----------------------------------------------------------------------------"
    )
    data_train, attributes, value_type = read(train_data_path, data_scheme_path)
    data_test, attr, value_type = read(test_data_path, data_scheme_path)

    # dataset = pre_process(data, attributes, value_type)

    manager = DatasetManager(data_train)

    # Generate rules
    start_time = perf_counter()
    cars = ga_genrules(
        manager,
        max_iter=50,
        pop_size=100,
        mutation_rate=0.5,
        min_support=minsup,
        min_confidence=minconf,
    )

    cars = sorted(cars, reverse=True)
    end_time = perf_counter()
    ga_rg_runtime = end_time - start_time

    # Bulding classifier
    start_time = perf_counter()
    classifier_m1 = m1_classifier_builder(cars, data_train)
    end_time = perf_counter()
    cb_runtime = end_time - start_time

    # Get accuracy score
    accuracy = classifier_m1.accuracy(data_test)

    # print("Cars:", len(cars.rules))
    print(f"Default class: {classifier_m1.default_class}")
    print(f"Number of rules: {len(classifier_m1.rules)}")
    print(classifier_m1.rules[-1])
    print(f"CBA-RG runtime is {ga_rg_runtime}")
    print(f"CBA-CB runtime is {cb_runtime}")
    print(f"Accuracy is {accuracy}")
    print("--------------------------------------------------------------------------")


if __name__ == "__main__":
    nursery_train = (
        "/home/haianh/grad_project/ml-learning/benchmark-dataset/nursery-train.data"
    )
    nursery_test = (
        "/home/haianh/grad_project/ml-learning/benchmark-dataset/nursery-test.data"
    )
    nursery_name = (
        "/home/haianh/grad_project/ml-learning/benchmark-dataset/nursery.names"
    )

    adult_train = "/home/haianh/grad_project/ml-learning/benchmark-dataset/adult-discretized-train.data"
    adult_test = "/home/haianh/grad_project/ml-learning/benchmark-dataset/adult-discretized-test.data"
    adult_name = "/home/haianh/grad_project/ml-learning/benchmark-dataset/adult.names"

    iris_train = "/home/haianh/grad_project/ml-learning/benchmark-dataset/iris-train.data"
    iris_test = "/home/haianh/grad_project/ml-learning/benchmark-dataset/iris-test.data"
    iris_name = "/home/haianh/grad_project/ml-learning/benchmark-dataset/iris.names"

    validate_with_dataset(nursery_train, nursery_test, nursery_name, 0.05, 0.5)
