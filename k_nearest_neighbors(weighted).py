import numpy as np
import pandas as pd

class MyKNNClf:
    def __init__(self, k=3, metric='euclidean', weight='uniform'):
        self.k = k
        self.X = None
        self.y = None
        self.metric = metric
        self.weight = weight
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
    
    def euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))
    
    def chebyshev_distance(self, point1, point2):
        return np.max(np.abs(point1 - point2))
    
    def manhattan_distance(self, point1, point2):
        return np.sum(np.abs(point1 - point2))
    
    def cosine_distance(self, point1, point2):
        return 1 - np.dot(point1, point2) / (np.linalg.norm(point1) * np.linalg.norm(point2))
    
    def _calculate_weights(self, distances):
        if self.weight == 'uniform':
            return np.ones(len(distances))
        elif self.weight == 'rank':
            return 1 / (np.arange(1, len(distances) + 1))
        elif self.weight == 'distance':
            distances = np.array(distances)
            return 1 / (distances)
    
    def predict(self, X: pd.DataFrame):
        predictions = []
        for i, test_point in X.iterrows():
            distances = []
            for j, train_point in self.X.iterrows():
                if self.metric == 'euclidean':
                    distance = self.euclidean_distance(test_point, train_point)
                elif self.metric == 'chebyshev':
                    distance = self.chebyshev_distance(test_point, train_point)
                elif self.metric == 'manhattan':
                    distance = self.manhattan_distance(test_point, train_point)
                elif self.metric == 'cosine':
                    distance = self.cosine_distance(test_point, train_point)
                distances.append((distance, self.y[j]))
            
            distances.sort(key=lambda x: x[0])
            nearest_neighbors = distances[:self.k]
            classes = np.array([neighbor[1] for neighbor in nearest_neighbors])
            weights = self._calculate_weights([neighbor[0] for neighbor in nearest_neighbors])
            
            if self.weight == 'uniform':
                prediction = max(set(classes), key=list(classes).count)
            else:
                weighted_votes = {}
                for cls, weight in zip(classes, weights):
                    if cls not in weighted_votes:
                        weighted_votes[cls] = 0
                    weighted_votes[cls] += weight
                prediction = max(weighted_votes, key=weighted_votes.get)
            
            predictions.append(prediction)
        return np.array(predictions)
    
    def predict_proba(self, X: pd.DataFrame):
        probabilities = []
        for i, test_point in X.iterrows():
            distances = []
            for j, train_point in self.X.iterrows():
                if self.metric == 'euclidean':
                    distance = self.euclidean_distance(test_point, train_point)
                elif self.metric == 'chebyshev':
                    distance = self.chebyshev_distance(test_point, train_point)
                elif self.metric == 'manhattan':
                    distance = self.manhattan_distance(test_point, train_point)
                elif self.metric == 'cosine':
                    distance = self.cosine_distance(test_point, train_point)
                distances.append((distance, self.y[j]))
            
            distances.sort(key=lambda x: x[0])
            nearest_neighbors = distances[:self.k]
            classes = np.array([neighbor[1] for neighbor in nearest_neighbors])
            weights = self._calculate_weights([neighbor[0] for neighbor in nearest_neighbors])
            
            if self.weight == 'uniform':
                proba = np.count_nonzero(classes == 1) / self.k
            else:
                weighted_sum = np.sum(weights[classes == 1])
                total_weight = np.sum(weights)
                proba = weighted_sum / total_weight
            
            probabilities.append(proba)
        return np.array(probabilities)
