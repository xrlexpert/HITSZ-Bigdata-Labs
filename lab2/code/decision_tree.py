import numpy as np
import pandas as pd
import random
cart_config = { 
    'max_depth': 10,  
    'min_samples_split': 2,  
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

def prune_tree(node, X_val, y_val):
    """
    对树进行后剪枝
    :param node: 当前节点
    :param X_val: 验证集特征
    :param y_val: 验证集标签
    :return: 剪枝后的节点
    """
    if y_val.shape[0] == 0:
        return node
    if node.label is not None:
        return node

    # 对左右子树进行递归剪枝
    if node.left:
        node.left = prune_tree(node.left, X_val[X_val[:, node.feature_index] == node.feature_value], y_val[X_val[:, node.feature_index] == node.feature_value])
    if node.right:
        node.right = prune_tree(node.right, X_val[X_val[:, node.feature_index] != node.feature_value], y_val[X_val[:, node.feature_index] != node.feature_value])

    # 评估当前节点是否需要剪枝
    if node.left is not None and node.right is not None:
        # 如果左右子节点都是叶子节点，则考虑剪枝
        if node.left.label is not None and node.right.label is not None:
            # 剪枝前后的准确率比较
            left_right_preds = np.where(X_val[:, node.feature_index] == node.feature_value, node.left.label, node.right.label)
            before_prune_acc = np.mean(left_right_preds == y_val)

            # 剪枝，直接将当前节点作为叶子节点
            node_preds = np.full(X_val.shape[0], np.argmax(np.bincount(y_val)))
            after_prune_acc = np.mean(node_preds == y_val)

            if after_prune_acc >= before_prune_acc:
                # 如果剪枝后准确率没有下降，则进行剪枝
                node = Node(None, None, None, None, np.argmax(np.bincount(y_val)))

    return node
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

if __name__ == "__main__":
    data = np.loadtxt('./data/train_adult_pro.csv', delimiter=',', skiprows=1)
    X,y = data[:,:-1], data[:,-1]
    X_train, y_train, X_val, y_val = split_train_val(X, y, ratio=0.8, seed=42)  # 划分训练集和验证集

    test_data = np.loadtxt('./data/test_adult_pro.csv', delimiter=',', skiprows=1)
    X_test, y_test =  test_data[:,:-1],test_data[:,-1]
    y_test = y_test.astype(int)

    #获取所有的切分点
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if((j,X[i][j]) in unused_splits):
                continue
            else:
                unused_splits.add((j,int(X[i][j])))

    cart = build_decision_tree(X_train, y_train, 0, unused_splits)
    cnt = 0
    for i in range(X_val.shape[0]):
        if(cart.predict(X_val[i]) == y_val[i]):
            cnt += 1
    print(f"before prune: test on val data:{cnt/X_val.shape[0]}")

    cnt = 0
    for i in range(X_test.shape[0]):
        if(cart.predict(X_test[i]) == y_test[i]):
            cnt += 1
    print(f"before prune: test on test data:{cnt/X_test.shape[0]}")
    
    # 后剪枝
    cart = prune_tree(cart, X_val, y_val)

    cnt = 0
    for i in range(X_val.shape[0]):
        if(cart.predict(X_val[i]) == y_val[i]):
            cnt += 1
    print(f"after prune: test on val data:{cnt/X_val.shape[0]}")




    cnt = 0
    for i in range(X_test.shape[0]):
        if(cart.predict(X_test[i]) == y_test[i]):
            cnt += 1
    print(f"after prune: test on test data:{cnt/X_test.shape[0]}")
    save_decision_tree(cart)