from apriori import apriori

if __name__ == "__main__":
    data_test_path = "/home/haianh/grad_project/ml-learning/association-rule-mining-apriori/data/groceries.csv"
    adult_new_data_path = "/home/haianh/grad_project/ml-learning/data/adult_new.data"
    nursery_data_path = "/home/haianh/grad_project/ml-learning/data/nursery.data"
    import csv

    def read(data_path: str):
        transactions = []
        items = []
        with open(data_path, "r") as f:
            reader = csv.reader(f)
            for line in reader:
                transactions.append(line)
                items.extend(line)
            unique_items = sorted(set(items))
        return transactions, unique_items

    

    # for testing agains result from efficient apriori package, grocery dataset
    transactions, items = read(nursery_data_path)

    # print(transactions)
    rules = apriori(transactions, min_support = 0.1, min_confidence=0.5)


    print(len(rules))
    # supports = [0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001]
    # num_itemsets = [8, 31, 63, 122, 333, 1001, 2226, 13492]

    # run_test(supports, result=num_itemsets)
