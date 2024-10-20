import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import random

columns = ['age', 'workclass', 'fnlwgt', 'education', 'educationNum', 'maritalStatus', 'occupation', 'relationship', 'race', 'sex',
          'capitalGain', 'capitalLoss', 'hoursPerWeek', 'nativeCountry', 'income']
df_train_set = pd.read_csv('./origin_data/adult.data', names=columns)
df_test_set = pd.read_csv('./origin_data/adult.test', names=columns, skiprows=1) #第一行是非法数据

print(len(df_train_set))
print(len(df_test_set))
df_train_set.to_csv('./origin_data/train_adult.csv', index=False)
df_test_set.to_csv('./origin_data/test_adult.csv', index=False)

df_train_set = pd.read_csv('./origin_data/train_adult.csv')
df_test_set = pd.read_csv('./origin_data/test_adult.csv')


df_train_set.drop(['fnlwgt', 'educationNum'], axis=1, inplace=True) # fnlwgt列用处不大，educationNum与education类似
df_test_set.drop(['fnlwgt', 'educationNum'], axis=1, inplace=True)
df_train_set.drop_duplicates(inplace=True) # 去除重复行
df_test_set.drop_duplicates(inplace=True)
df_train_set.dropna(inplace=True) # 去除空行 
df_test_set.dropna(inplace=True)

# 去除含有'?'的行
new_columns = ['workclass', 'education', 'maritalStatus', 'occupation', 'relationship', 'race', 'sex',
               'nativeCountry', 'income']
for col in new_columns:
    df_train_set = df_train_set[~df_train_set[col].str.contains(r'\?', regex=True)]
    df_test_set = df_test_set[~df_test_set[col].str.contains(r'\?', regex=True)]
#save to csv
df_train_set.to_csv('./origin_data/train_adult.csv', index=False)
df_test_set.to_csv('./origin_data/test_adult.csv', index=False)

#连续变量离散化
continuous_column = ['age', 'capitalGain', 'capitalLoss', 'hoursPerWeek']
allbins = [[0, 20, 40, 60, 80,100], 
        [0, 10000, 50000, 100000],
        [0,1,5000], 
        [0, 20, 40, 60, 80, 100]]
for col, bins in zip(continuous_column, allbins):
    df_train_set[col] = pd.cut(df_train_set[col], bins, right=False, labels=False)
    df_test_set[col] = pd.cut(df_test_set[col], bins, right=False, labels=False)

print(df_train_set.head())  
print(df_test_set.head())       


# #离散变量index化
discrete_column = ['workclass', 'education', 'maritalStatus', 'occupation', 'relationship', 'race', 'sex', 'nativeCountry', 'income']
workclass_mapping = {' Private': 0, ' Self-emp-not-inc': 1, ' Self-emp-inc': 1, ' Local-gov': 2, 
                     ' State-gov': 2, ' Federal-gov': 2, ' Without-pay': 3, ' Never-worked': 3}
education_mapping = {
    ' 5th-6th': 0,
    ' 7th-8th': 0,
    ' 9th': 0,
    ' 10th': 0,
    ' 11th': 0,
    ' 12th': 0,
    ' HS-grad': 0,
    ' Preschool': 1,
    ' 1st-4th': 1,
    ' Assoc-acdm': 2,
    ' Assoc-voc': 2,
    ' Some-college': 3,
    ' Bachelors': 3,
    ' Doctorate': 4,
    ' Prof-school': 4,
    ' Masters': 4
}

income_mapping = {' <=50K': 0, ' <=50K.':0, ' >50K': 1, ' >50K.': 1}
special_mapping_name = ['workclass', 'education' 'income']
df_test_set['workclass'] = df_test_set['workclass'].map(workclass_mapping)
df_train_set['workclass'] = df_train_set['workclass'].map(workclass_mapping)
df_test_set['education'] = df_test_set['education'].map(education_mapping)
df_train_set['education'] = df_train_set['education'].map(education_mapping)
df_test_set['income'] = df_test_set['income'].map(income_mapping)
df_train_set['income'] = df_train_set['income'].map(income_mapping)
print(df_train_set.head())
print(df_test_set.head())
for col in discrete_column:
    if(col in special_mapping_name):
      continue
    else:
        res1 = df_test_set[col].value_counts().keys()
        res2 = df_train_set[col].value_counts().keys()
        res = list(set(res1).union(set(res2)))
        mapping = dict(zip(res, range(len(res))))
        print(mapping)
        df_train_set[col] = df_train_set[col].map(mapping)
        df_test_set[col] = df_test_set[col].map(mapping)
print(df_train_set.head())
print(df_test_set.head())
#save to csv
df_train_set.to_csv('./data/train_adult_pro.csv', index=False)
df_test_set.to_csv('./data/test_adult_pro.csv', index=False)        
       
