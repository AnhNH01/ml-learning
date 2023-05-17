from itemset import generate_frequent_itemsets
from arules import generate_association_rules
from utils import timeit
from time import perf_counter

def apriori(transaction, min_support=0.1, min_confidence=0.5, max_length=15):
    frequent_itemsets, transaction_num = generate_frequent_itemsets(
        transaction, min_support, max_length
    )

    return generate_association_rules(
        frequent_itemsets, min_confidence, transaction_num
    )


if __name__ == "__main__":
    data_test_path = "/home/haianh/grad_project/ml-learning/association-rule-mining-apriori/data/groceries.csv"

    from utils import read

    # for testing agains result from efficient apriori package, grocery dataset
    transactions, items = read(data_test_path)

    SUPPORTS = [0.05, 0.03, 0.01, 0.005, 0.003]
    CONFIDENCES = [0.5, 0.7, 0.9]
    EXPECTED_RESULT = {
        (0, 0): 0,
        (0, 1): 0,
        (0, 2): 0,
        (1, 0): 0,
        (1, 1): 0,
        (1, 2): 0,
        (2, 0): 15,
        (2, 1): 0,
        (2, 2): 0,
        (3, 0): 120,
        (3, 1): 1,
        (3, 2): 0,
        (4, 0): 422,
        (4, 1): 19,
        (4, 2): 0,
    }

    def run_test(supports: list, confidence: list, expected: dict[tuple, int]):
        test_result: dict[tuple, tuple[int, float]] = dict()
        for i, sup in enumerate(supports):
            for j, conf in enumerate(confidence):
                s = perf_counter()
                rules = apriori(transactions, min_support=sup, min_confidence=conf)
                e = perf_counter()

                test_result[(i, j)] = (len(rules), e-s)

        for index, rsl in enumerate(test_result):
            if test_result[rsl][0] == expected[rsl]:
                print(
                    f" Test {index+1} --> PASSED: {test_result[rsl][0]} rules at {supports[rsl[0]]} minimum support and {confidence[rsl[1]]} min cofidence, Time: {test_result[rsl][1]} seconds"
                )
            else:
                print(
                    f" Test {index+1} --> FAILED: {expected[rsl]} rules expected, got {test_result[rsl][0]} at {supports[rsl[0]]} minimum support and {confidence[rsl[1]]} min cofidence, Time: {test_result[rsl][1]} seconds"
                )

    # print("____________Test result on Grocery Data______________")
    # run_test(SUPPORTS, CONFIDENCES, EXPECTED_RESULT)
    # print("______________________________________________________________")

    adult_cleaned_data = "/home/haianh/grad_project/ml-learning/data/adult_new.data"
    transactions,items = read(adult_cleaned_data)
    
    adult_cleaned_data = "/home/haianh/grad_project/ml-learning/data/adult_new.data"
    transactions,items = read(adult_cleaned_data)
    
    # rules = apriori(transactions, min_support=0.03,min_confidence=0.7, max_length=8)
    # print(len(rules))
    test_result = dict()
    supports = [0.05, 0.03, 0.01]
    confidences = [0.5, 0.7, 0.9]
    for (i,sup) in enumerate(supports):
        for (j,conf) in enumerate(confidences):
            s = perf_counter()
            rules = apriori(transactions, min_support=sup, min_confidence=conf, max_length=8)
            e = perf_counter()
            test_result[(supports[i], confidences[j])] = (len(rules),e - s)

    for key in test_result:
        print(f'{key}: {test_result[key][0]} after {test_result[key][1]}')