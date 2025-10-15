import numpy as np
import pandas as pd

class MyTreeClf:
    def __init__(self, max_depth=5, min_samples_split=2, max_leafs=20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.tree = None
        self.leafs_cnt = 0

    def node_entropy(self, probs):
        return -np.sum([p * np.log2(p) for p in probs if p > 0])

    def node_ig(self, x_col, y, split_value):
        left_mask = x_col <= split_value
        right_mask = x_col > split_value

        if len(x_col[left_mask]) == 0 or len(x_col[right_mask]) == 0:
            return 0

        left_counts = np.bincount(y[left_mask])
        right_counts = np.bincount(y[right_mask])

        left_probs = left_counts / len(y[left_mask]) if len(y[left_mask]) > 0 else np.zeros_like(left_counts)
        right_probs = right_counts / len(y[right_mask]) if len(y[right_mask]) > 0 else np.zeros_like(right_counts)

        entropy_after = (len(y[left_mask]) / len(y) * self.node_entropy(left_probs) +
                         len(y[right_mask]) / len(y) * self.node_entropy(right_probs))
        entropy_before = self.node_entropy(np.bincount(y) / len(y))

        return entropy_before - entropy_after

    def get_best_split(self, X: pd.DataFrame, y: pd.Series):
        best_col, best_split_value, best_ig = None, None, -np.inf

        for col in X.columns:
            sorted_unique_values = np.sort(X[col].unique())

            for i in range(1, len(sorted_unique_values)):
                split_value = (sorted_unique_values[i - 1] + sorted_unique_values[i]) / 2

                ig = self.node_ig(X[col], y, split_value)

                if ig > best_ig:
                    best_ig = ig
                    best_col = col
                    best_split_value = split_value

        return best_col, best_split_value

    def fit(self, X: pd.DataFrame, y: pd.Series, depth=1, node=None):
        if self.max_leafs < 2:
            self.leafs_cnt = 2
        
        if node is None:
            node = {}
            self.tree = node
            
        best_col, best_split_value = self.get_best_split(X, y)

        node['type'] = None
        node['feature'] = best_col
        node['threshold'] = best_split_value

        tmp_leafs = 2
        
        if len(y.unique()) == 1:
            self.leafs_cnt += 1
            tmp_leafs -= 1
            node['type'] = 'leaf'
            node['class_counts'] = dict(zip(*np.unique(y, return_counts=True)))
            return

        if len(y) == 1:
            self.leafs_cnt += 1
            tmp_leafs -= 1
            node['type'] = 'leaf'
            node['class_counts'] = dict(zip(*np.unique(y, return_counts=True)))
            return
        
        if depth > self.max_depth or len(y) < self.min_samples_split or (tmp_leafs + self.leafs_cnt) > self.max_leafs:
            self.leafs_cnt += 1
            tmp_leafs -= 1
            node['type'] = 'leaf'
            node['class_counts'] = dict(zip(*np.unique(y, return_counts=True)))
            return
        
        if best_col is None:
            self.leafs_cnt += 1
            tmp_leafs -= 1
            node['type'] = 'leaf'
            node['class_counts'] = dict(zip(*np.unique(y, return_counts=True)))
            return

        node['type'] = 'node'
        node['feature'] = best_col
        node['threshold'] = best_split_value

        tmp_leafs += 2

        left_mask = X[best_col] <= best_split_value
        right_mask = X[best_col] > best_split_value

        node['left'] = {}
        node['right'] = {}

        self.fit(X[left_mask], y[left_mask], depth + 1, node['left'])
        self.fit(X[right_mask], y[right_mask], depth + 1, node['right'])

    def bypass_tree(self, node, sample):
        while node['type'] == 'node':
            feature_value = sample[node['feature']]
            if feature_value <= node['threshold']:
                node = node['left']
            else:
                node = node['right']

        if node['type'] == 'leaf':
            return node

    def predict(self, X: pd.DataFrame):
        classification = []
        proba = self.predict_proba(X)
        for p in proba:
            if p > 0.5:
                classification.append(1)
            else:
                classification.append(0)

        return classification

    def predict_proba(self, X: pd.DataFrame):
        proba = []

        for _, sample in X.iterrows():
            node = self.bypass_tree(self.tree, sample)
            if node:
                overall_cnt = sum(node['class_counts'].values())
                class1_cnt = node['class_counts'].get(1, 0)
                class1_proba = (class1_cnt / overall_cnt) if overall_cnt > 0 else 0.0
    
                proba.append(class1_proba)
    
        return np.array(proba)
