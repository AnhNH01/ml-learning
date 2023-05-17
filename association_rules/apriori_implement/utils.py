import csv
from time import perf_counter
import functools

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


def timeit(func):
    @functools.wraps(func)
    def wrapper(*arg, **kwargs):
        st = perf_counter()
        value = func(*arg, **kwargs)
        end = perf_counter()
        print(f"{func.__name__} executed in {end-st} seconds")
        return value
    return wrapper
