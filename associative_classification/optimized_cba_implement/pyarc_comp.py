from sklearn.model_selection import train_test_split
import pandas as pd
from pyarc import TransactionDB,CBA
from time import perf_counter


def validate(train_path, test_path, minsup, minconf):
    print(f"************Pyarc at {minsup} support and {minconf} confidence")
    print("--------------------------------------------------------------")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    txns_train = TransactionDB.from_DataFrame(train)
    txns_test = TransactionDB.from_DataFrame(test)

    cba = CBA(support=minsup, confidence=minconf, algorithm="m1")
    s = perf_counter()
    cba.fit(txns_train)
    e = perf_counter()
    
    accuracy = cba.rule_model_accuracy(txns_test) 

    print("Number of rules in classifier: ", len(cba.clf.rules))

    print("Default class:", cba.clf.default_class)
    # print(cba.clf.rules[0])
    # for rule in cba.clf.rules:
    #     print(rule)
    print(f"Run time is {e - s} seconds")
    print(f"Accuracy: {accuracy * 100}")
    print("--------------------------------------------------------------")


nursery_train_path = "/home/haianh/grad_project/ml-learning/benchmark-dataset/nursery-pyarc-train.data"
nursery_test_path = "/home/haianh/grad_project/ml-learning/benchmark-dataset/nursery-pyarc-test.data"

adult_train_path = "/home/haianh/grad_project/ml-learning/benchmark-dataset/adult-discretized-pyarc-train.data"
adult_test_path = "/home/haianh/grad_project/ml-learning/benchmark-dataset/adult-discretized-pyarc-test.data"

pen_train = "/home/haianh/grad_project/ml-learning/data/pen.tra"
pen_test = "/home/haianh/grad_project/ml-learning/data/pen.test"

letter_train = "/home/haianh/grad_project/ml-learning/benchmark-dataset/letter-pyarc-train.data"
letter_test = "/home/haianh/grad_project/ml-learning/benchmark-dataset/letter-pyarc-test.data"

krkopt_train = "/home/haianh/grad_project/ml-learning/benchmark-dataset/krkopt-train.data"
krkopt_test = "/home/haianh/grad_project/ml-learning/benchmark-dataset/krkopt-test.data"

connect4_train = "/home/haianh/grad_project/ml-learning/benchmark-dataset/connect4-train.data"
connect4_test = "/home/haianh/grad_project/ml-learning/benchmark-dataset/connect4-test.data"


android = "/home/haianh/grad_project/ml-learning/data/data.csv"

student = "/home/haianh/grad_project/ml-learning/data/student_data.csv"

coupon_train = "/home/haianh/grad_project/ml-learning/benchmark-dataset/coupon-pyarc.train"
coupon_test = "/home/haianh/grad_project/ml-learning/benchmark-dataset/coupon-pyarc.test"

mushroom_train = "/home/haianh/grad_project/ml-learning/benchmark-dataset/mushroom-pyarc.train"
mushroom_test = "/home/haianh/grad_project/ml-learning/benchmark-dataset/mushroom-pyarc.test"

validate(mushroom_train, mushroom_test, 0.05, 0.5)

