from read import read
from pre_processing import pre_process
from time import perf_counter
from ga_rg import ga_genrules
from m1_algo import m1_classifier_builder
from manager import DatasetManager


def validate_with_dataset(
    train_data_path, test_data_path, data_scheme_path, minsup=0.01, minconf=0.5
):
    print(f"         GAAC at {minsup} support and {minconf} confidence ")
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
        pop_size=150,
        mutation_rate=0.1,
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

    print("Cars:", len(cars))
    print(f"Default class: {classifier_m1.default_class}")
    print(f"Number of rules: {len(classifier_m1.rules)}")
    
    print(f"GA-RG runtime is {ga_rg_runtime}")
    print(f"CB runtime is {cb_runtime}")
    print(f"Accuracy is {accuracy}")
    print(f"Rule data-train coverage: {manager.get_dataset_coverage(classifier_m1.rules)}")
    # classifier_m1.print()
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

    pen_train = "/home/haianh/grad_project/ml-learning/data/pen.tra"
    pen_test = "/home/haianh/grad_project/ml-learning/data/pen.test"
    pen_name = "/home/haianh/grad_project/ml-learning/data/pen.names"

    letter_train = "/home/haianh/grad_project/ml-learning/benchmark-dataset/letter-train.data"
    letter_test = "/home/haianh/grad_project/ml-learning/benchmark-dataset/letter-test.data"
    letter_name =  "/home/haianh/grad_project/ml-learning/benchmark-dataset/letter.names"

    bank_train = "/home/haianh/grad_project/ml-learning/benchmark-dataset/bank-train.data"
    bank_test = "/home/haianh/grad_project/ml-learning/benchmark-dataset/bank-test.data"
    bank_name = "/home/haianh/grad_project/ml-learning/benchmark-dataset/bank.names"

    krkopt_train = "/home/haianh/grad_project/ml-learning/benchmark-dataset/krkopt-train.data"
    krkopt_test = "/home/haianh/grad_project/ml-learning/benchmark-dataset/krkopt-test.data"
    krkopt_name = "/home/haianh/grad_project/ml-learning/benchmark-dataset/krkopt.names"

    connect4_train = "/home/haianh/grad_project/ml-learning/benchmark-dataset/connect4-train.data"
    connect4_test = "/home/haianh/grad_project/ml-learning/benchmark-dataset/connect4-test.data"
    connect4_name = "/home/haianh/grad_project/ml-learning/benchmark-dataset/connect4.names"

    coupon_train = "/home/haianh/grad_project/ml-learning/benchmark-dataset/coupon-train.data"
    coupon_test = "/home/haianh/grad_project/ml-learning/benchmark-dataset/coupon-test.data"
    coupon_name = "/home/haianh/grad_project/ml-learning/benchmark-dataset/coupon.names"

    mushroom_train = "/home/haianh/grad_project/ml-learning/benchmark-dataset/mushroom-train.data"
    mushroom_test = "/home/haianh/grad_project/ml-learning/benchmark-dataset/mushroom-test.data"
    mushroom_name = "/home/haianh/grad_project/ml-learning/benchmark-dataset/mushroom.names"

    adult_full_train = '/home/haianh/grad_project/ml-learning/benchmark-dataset/adult-real-train.data'
    adult_full_test = '/home/haianh/grad_project/ml-learning/benchmark-dataset/adult-real-test.data'

    # supports = [0.05, 0.03, 0.01, 0.005, 0.003]
    # for sup in supports:
    #     for _ in range(0, 5):
    #         validate_with_dataset(bank_train, bank_test, bank_name, sup, 0.5)
    #         print()
    #     print()
    #     print()
    validate_with_dataset(bank_train, bank_test, bank_name, 0.2, 0.5)