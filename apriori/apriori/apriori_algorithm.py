# -*- coding: utf-8 -*-
"""
http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/
http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori


dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
           ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
           ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]

te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)

print frequent_itemsets

#The generate_rules() function allows you to (1) specify your metric of interest and 
#(2) the according threshold. Currently implemented measures are confidence and lift. 
from mlxtend.frequent_patterns import association_rules

association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

#Now, let us return the items and itemsets with at least 60% support:
from mlxtend.frequent_patterns import apriori


#By default, apriori returns the column indices of the items, 
#which may be useful in downstream operations such as association rule mining.
# For better readability, we can set use_colnames=True 
#to convert these integer values into the respective item names: 
apriori(df, min_support=0.6, use_colnames=True)

#Selecting and Filtering Results
#let's assume we are only interested in itemsets of length 2 
#that have a support of at least 80 percent. 
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets
#Then, we can select the results that satisfy our desired criteria as follows:
frequent_itemsets[ (frequent_itemsets['length'] == 2) &
                   (frequent_itemsets['support'] >= 0.8) ]
#Similarly, using the Pandas API, we can select entries based on the "itemsets" column: 
frequent_itemsets[ frequent_itemsets['itemsets'] == {'Onion', 'Eggs'} ]                   
                   
                 