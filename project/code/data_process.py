import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


train = pd.read_csv('./data/train.csv')
test  = pd.read_csv('./data/test.csv')

test_id = test['Id']

train.drop(['Id'], axis=1, inplace=True)

continuous_columns = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
need_to_remove = ['MSSubClass','MoSold', 'YrSold','GarageYrBlt','SalePrice']
for col in need_to_remove:
    continuous_columns.remove(col)

discret_columns = train.select_dtypes(include=['object']).columns.tolist()
need_to_add = ['MSSubClass','MoSold', 'YrSold','GarageYrBlt']
for col in need_to_add:
    discret_columns.append(col)

for col in continuous_columns:
    train[col] = train[col].fillna(train[col].median())
    test[col] = test[col].fillna(test[col].median())

for col in discret_columns:
    train[col].fillna(-1,inplace=True)
    test[col].fillna(-1,inplace=True)
    res1 = test[col].value_counts().keys()
    res2 = train[col].value_counts().keys()
    res = list(set(res1).union(set(res2)))
    mapping = dict(zip(res, range(len(res))))
    print(f'{col}:{mapping}')
    train[col] = train[col].map(mapping)
    test[col] = test[col].map(mapping)

train.to_csv('./data/train_pro.csv',index=False)
test.to_csv('./data/test_pro.csv',index=False)
