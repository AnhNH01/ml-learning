from typing import Hashable, Iterable
from itertools import combinations
from manager import TransactionManager


def join_step(k_itemsets: Iterable[tuple]):
    """
    Join step in apriori paper

    Used to generate k+1 length itemset from list of k length itemset

    Params:

        k_itemsets: list of k length itemset

    Examples

    itemsets = [(1, 2, 3), (1, 2, 4), (1, 3, 4), (1, 3, 5), (2, 3, 4)]

    list(join_step(itemsets))

    [(1, 2, 3, 4), (1, 3, 4, 5)]

    """

    i = 0
    while i < len(k_itemsets):
        *first, last = k_itemsets[i]
        skip = 1
        tail_items = [last]

        for j in range(i + 1, len(k_itemsets)):
            *first_items, last_item = k_itemsets[j]

            if first == first_items:
                tail_items.append(last_item)
                skip += 1

            else:
                break

        for combination in combinations(tail_items, 2):
            yield tuple(sorted(first + list(combination)))

        i += skip


def prune_step(k_itemsets: Iterable[tuple], possible_candidates: Iterable[tuple]):
    """
    Prune candidate itemset generated in join_step
    by removing candidate which subset not in previous itemset
    (in original paper)

    Params:

    k_itemsets: k length itemsets

    possible_candidates: k+1 length itemsets generated in the join step


    """

    for candidate in possible_candidates:
        # doesnt need to check subset created by removing last 2 item because they must be in the k_itemsets
        for i in range(0, len(candidate) - 2):
            # subset created by removing the i-th item in the itemset
            subset = candidate[:i] + candidate[i + 1 :]

            if subset not in k_itemsets:
                break
        else:
            yield candidate


def apriori_gen(k_itemsets: Iterable[tuple]):
    """
    Generate k+1 itemset candidates from k_itemsets

    2 phases: Join step and prune step.

    join_step: create join k_itemsets to create all possible k+1_itemsets

    prune_step: prune all k+1_itemsets which k_subset does not exist in k_itemset
    due to property:
        Support(Superset) > k --> Support(subset) > k for every subset of Superset
    """

    candidate = join_step(k_itemsets)
    yield from prune_step(k_itemsets, candidate)


def generate_frequent_itemsets(transactions, min_support=0.1, max_length=15) -> tuple[dict[int, dict[tuple, int]], int]:
    manager = TransactionManager(transactions)

    frequent_itemsets: dict[int, dict[tuple, int]] = dict()

    itemset_length = 1

    item_over_min_support = [
        item
        for item in manager.items
        if len(manager.transaction_indices_by_item[item]) / manager.transaction_num
        >= min_support
    ]

    # create 1-large-itemset
    one_large_itemset = {
        (item,): len(manager.transaction_indices_by_item[item])
        for item in item_over_min_support
    }

    # add to frequent_itemsets
    frequent_itemsets[itemset_length] = dict(sorted(one_large_itemset.items()))
    
    itemset_length = 2
    while len(frequent_itemsets[itemset_length - 1].keys()) and max_length > 1:
        itemset_list = list(frequent_itemsets[itemset_length - 1].keys())

        possible_candidate_itemsets = list(apriori_gen(itemset_list))

        candidate_itemsets: dict[tuple, int] = dict() 
        for candidate in possible_candidate_itemsets:
            candidate_count = len(manager.transaction_index(candidate))
            candidate_support = (
                candidate_count / manager.transaction_num
            )
            if candidate_support >= min_support:
                candidate_itemsets[candidate] = candidate_count

        if not candidate_itemsets:
            break

        frequent_itemsets[itemset_length] = dict(sorted(candidate_itemsets.items()))
        itemset_length += 1
        if itemset_length > max_length:
            break

    return frequent_itemsets, manager.transaction_num


if __name__ == "__main__":
    data_test_path = "/home/haianh/grad_project/ml-learning/association-rule-mining-apriori/data/groceries.csv"
    adult_new_data = "/home/haianh/grad_project/ml-learning/data/adult_new.data"
    from utils import read
    def run_test(supports: list, result: list):
        test_result = []
        for sup in supports:
            fi, _ = generate_frequent_itemsets(transactions, min_support=sup, max_length=8)

            count = 0
            for i in fi:
                count += len(fi[i])
            test_result.append(count)

        for i in range(len(supports)):
            if result[i] == test_result[i]:
                print(
                    f" Test {i+1} --> PASSED: {test_result[i]} itemsets at {supports[i]} minimum support"
                )
            else:
                print(
                    f" Test {i+1} --> FAILED: {result[i]} itemsets expected, got {test_result[i]} at {supports[i]} minimum support"
                )

    # for testing agains result from efficient apriori package, grocery dataset
    # transactions, items = read(data_test_path)
    # supports = [0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001]
    # num_itemsets = [8, 31, 63, 122, 333, 1001, 2226, 13492]

    supports = [0.05, 0.03, 0.01]
    num_itemsets = [5306, 11687, 53703]
    transactions, items = read(adult_new_data)
    run_test(supports, result=num_itemsets)

