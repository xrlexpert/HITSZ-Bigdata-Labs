import numpy as np
import pandas as pd
import random
cart_config = { 
    'max_depth': 10,  
    'min_samples_split': 1,  
    'min_gini': 0.02
}
unused_splits = set()  # 记录已经使用过的切分点，避免重复使用

class Node:
    def __init__(self, feature_index, feature_value, left, right, label):
        self.feature_index = feature_index
        self.feature_value = feature_value
        self.left = left
        self.right = right
        self.label = label

    def predict(self, x):
        if self.label is not None:
            return self.label
        if x[self.feature_index] == self.feature_value:
            return self.left.predict(x)
        else:
            return self.right.predict(x)

    def __str__(self):
        return 'feature_index: %d, feature_value: %d, label: %d' % (self.feature_index, self.feature_value, self.label)

def calc_gini(y):
    """
    计算数据集的基尼指数
    :param X: 当前节点所包含数据
    :return: 基尼指数
    """
    n = len(y)
    if n == 0:
        return 0
    m = y.sum()
    prob = m / n
    gini = 2 * prob * (1 - prob)
    return gini

def split_dataset(X, y,feature_index, feature_value):
    """
    按照给定的列划分数据集
    :param X: 当前节点所包含数据
    :param index: 指定特征的列索引
    :param value: 指定特征的值
    :return: 切分后的数据集
    """
    left = X[X[:, feature_index] == feature_value]
    right = X[X[:, feature_index] != feature_value]
    left_labels = y[X[:, feature_index] == feature_value]
    right_labels = y[X[:, feature_index] != feature_value]
    return left, right, left_labels, right_labels

    
    
def choose_best_feature_to_split(X, y):
    """
    选择最好的特征进行分裂
    :param X: 当前节点数据
    :return: best_value:(分裂特征的index, 特征的值), best_df:(分裂后的左右子树数据集), best_gain:(选择该属性分裂的最大信息增益)
    """
    best_gini = 1
    best_feature = -1
    best_split = None
    best_value = -1
    n = X.shape[0]
    for (i, j) in unused_splits:
        left, right, left_labels, right_labels = split_dataset(X, y, i, j)
        gini = left.shape[0]/n * calc_gini(left_labels) + right.shape[0]/n * calc_gini(right_labels)
        if gini < best_gini:
            best_gini = gini
            best_feature = i
            best_value = j
            best_split = (left, right, left_labels, right_labels)
    return best_feature, best_value, best_split, best_gini


def build_decision_tree(X, y, depth, flags):
    """
    构建CART树
    :param X: 数据集
    :param y: 标签集
    :param depth: 当前深度
    :return: CART树
    """
    if(len(np.unique(y)) == 1):
        return Node(None, None, None, None,np.argmax(np.bincount(y)))
    if depth >= cart_config['max_depth']:
        return Node(None, None, None, None, np.argmax(np.bincount(y)))
    if y.shape[0] <= cart_config['min_samples_split']:
        return Node(None, None, None, None, np.argmax(np.bincount(y)))
    gini = calc_gini(y)
    if(gini <= cart_config['min_gini']):
        return Node(None, None, None, None, np.argmax(np.bincount(y)))
    if(len(unused_splits) == 0):
        return Node(None, None, None, None, np.argmax(np.bincount(y)))
    best_feature, best_value, best_split, best_gini = choose_best_feature_to_split(X, y)
    if(best_gini >= gini):
        return Node(None, None, None, None, np.argmax(np.bincount(y)))
    node = Node(best_feature, best_value, None, None, None)
    node.left = build_decision_tree(best_split[0], best_split[2], depth + 1, flags)
    node.right = build_decision_tree(best_split[1], best_split[3], depth + 1, flags)
    return node

    
    
def save_decision_tree(cart):
    """
    决策树的存储
    :param cart: 训练好的决策树
    :return: void
    """
    np.save('cart.npy', cart)
    
    
def load_decision_tree():
    """
    决策树的加载
    :return: 保存的决策树
    """    
    
    cart = np.load('cart.npy', allow_pickle=True)
    return cart.item()


if __name__ == "__main__":
    train_data = np.loadtxt('train_adult_pro.csv', delimiter=',', skiprows=1)
    X,y = train_data[:,:-1],train_data[:,-1]
    y = y.astype(int)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if((j,X[i][j]) in unused_splits):
                continue
            else:
                unused_splits.add((j,int(X[i][j])))
    cart = build_decision_tree(X, y, 0, unused_splits)
    save_decision_tree(cart)
    cart = load_decision_tree()
    cnt = 0
    test_data = np.loadtxt('test_adult_pro.csv', delimiter=',', skiprows=1)
    for i in range(X.shape[0]):
        if(cart.predict(X[i]) == y[i]):
            cnt += 1
    print(f"test on train data:{cnt/X.shape[0]}")
    cnt = 0
    X_test, y_test =  test_data[:,:-1],test_data[:,-1]
    y_test = y_test.astype(int)
    for i in range(X_test.shape[0]):
        if(cart.predict(X_test[i]) == y_test[i]):
            cnt += 1
    print(f"test on test data:{cnt/X_test.shape[0]}")
