from collections import defaultdict
from typing import Hashable, Iterable
import collections.abc

class TransactionManager:   
    """
    Manager for transactions and items

    transactions: list[list|set|tuple] --> list of transactions
        
        eg: [
            [1, 2, 4],
            [1 , 4],
            [2, 3]
        ]
    
    
    transaction_indices_by_item: defaultdict[set] --> contain indices of transactions containing that item

        eg:
        {
            1: {0, 1}, #index of transaction containing 1

            2: {0, 2},

            3: {2},

            4: {0, 1}

        }
    
    transaction_num: total number of transactions


    """
    
    def __init__(self, transactions:Iterable[Iterable[Hashable]]):
        
        self.transaction_indices_by_item = defaultdict(set)
        self.transaction_num = len(transactions)

        for transaction_index, transaction in enumerate(transactions):
            for item in transaction:
                self.transaction_indices_by_item[item].add(transaction_index)
        
    @property
    def items(self):
        """
        Return list of items
        """
        return self.transaction_indices_by_item.keys()
        
    def transaction_index(self, transaction:Iterable[Hashable]):
        """
        Return index of transaction by intersecting indices of items in transaction_indices_by_item

        eg: 
        
        transaction_index([2,3])

            transaction_indices_by_item['2'] = {0, 2}
            transaction_indices_by_item['3'] = {2}

            {0, 2} intersect {2} = 2
            --> index of transaction [2, 3] is 2

        """
        
        transaction_cpy = set(transaction)
        item = transaction_cpy.pop()
        indices:set = self.transaction_indices_by_item[item]

        while transaction_cpy:
            item = transaction_cpy.pop()
            indices = self.transaction_indices_by_item[item].intersection(indices)
        
        return indices
    

