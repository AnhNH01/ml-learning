from sklearn.model_selection import train_test_split
import pandas as pd
from pyarc import TransactionDB,CBA

# data_path = "/home/haianh/grad_project/ml-learning/data/py_arc_dresses.data"
# data_path = "/home/haianh/grad_project/ml-learning/data/adult_pyarc.data"
# data_path = "/home/haianh/grad_project/ml-learning/data/iris.csv"
data_path = "/home/haianh/grad_project/ml-learning/data/nursery_pyarc.data"
data_frame = pd.read_csv(data_path)

# data_train, data_test = train_test_split(data_frame, shuffle=True, test_size=0.2)
txns_train = TransactionDB.from_DataFrame(data_frame)
# txns_test = TransactionDB.from_DataFrame(data_test)
cba = CBA(support=0.05, confidence=0.5, algorithm="m1")


cba.fit(txns_train)

# accuracy = cba.rule_model_accuracy(txns_test) 

print("Number of rules in classifier: ", len(cba.clf.rules))

print("Default class:", cba.clf.default_class)
# print(cba.clf.rules)

# print(accuracy)