import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import sklearn


def split_train_val(X, y, ratio=0.8, seed=42):
    """
    划分训练集和验证集
    :param X: 特征
    :param y: 标签
    :param ratio: 训练集比例
    :return: X_train, y_train, X_val, y_val
    """
    np.random.seed(seed)
    n = X.shape[0]
    y = y.astype(int)
    indices = np.arange(n)
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]
    split = int(n * ratio)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    return X_train, y_train, X_val, y_val

config={
    'ratio':0.8,
    'seed':42
}
df = pd.read_csv('./data/train_pro.csv')
data = df.values
n = data.shape[0]
X = data[:,:-1]
y = data[:,-1]

X_train, y_train, X_val, y_val = split_train_val(X, y, config.get('ratio'), seed=config.get('seed'))
#choose the best model
xgb_model = XGBRegressor(learning_rate=0.015,n_estimators=4750,max_depth=3,min_child_weight=0,subsample=0.7,colsample_bytree=0.4064,nthread=-1,scale_pos_weight=2,seed=42)
xgb_model.fit(X_train,y_train, eval_set = [(X_val,y_val)])


df_test = pd.read_csv('./data/test_pro.csv')
data_test = df_test.values
X_test = data_test[:,1:]
pred = xgb_model.predict(X_test)
pred_train = xgb_model.predict(X_train)
submission = pd.DataFrame({
    'Id': data_test[:,0].astype(int),  # Assuming test set has an 'Id' column
    'SalePrice': pred # Replace 'Target' with the name of your target column
})

# Save the submission file
submission.to_csv('./res/submission.csv', index=False)


