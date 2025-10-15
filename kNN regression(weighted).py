import numpy as np
import pandas as pd

class MyKNNReg:
    def __init__(self, k=3, metric='euclidean', weight='uniform'):
        self.k = k
        self.metric = metric
        self.weight = weight
        self.X_train = None
        self.y_train = None

    def euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def chebyshev_distance(self, point1, point2):
        return np.max(np.abs(point1 - point2))

    def manhattan_distance(self, point1, point2):
        return np.sum(np.abs(point1 - point2))

    def cosine_distance(self, point1, point2):
        return 1 - np.dot(point1, point2) / (np.linalg.norm(point1) * np.linalg.norm(point2))

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X):
        predictions = []
        for i, test_point in X.iterrows():
            distances = []
            for j, train_point in self.X_train.iterrows():
                if self.metric == 'euclidean':
                    distance = self.euclidean_distance(test_point, train_point)

                elif self.metric == 'chebyshev':
                    distance = self.chebyshev_distance(test_point, train_point)

                elif self.metric == 'manhattan':
                    distance = self.manhattan_distance(test_point, train_point)

                elif self.metric == 'cosine':
                    distance = self.cosine_distance(test_point, train_point)
                    
                distances.append((distance, self.y_train[j]))

            distances.sort(key=lambda x: x[0])
            nearest_neighbors = distances[:self.k]

            if self.weight == 'uniform':
                prediction = np.mean([neighbor[1] for neighbor in nearest_neighbors])

            elif self.weight == 'distance':
                weighted_sum = sum(neighbor[1] / neighbor[0] for neighbor in nearest_neighbors if neighbor[0] != 0)
                total_weight = sum(1 / neighbor[0] for neighbor in nearest_neighbors if neighbor[0] != 0)

                if total_weight == 0:
                    prediction = np.mean([neighbor[1] for neighbor in nearest_neighbors])

                else:
                    prediction = weighted_sum / total_weight

            elif self.weight == 'rank':
                rank_weights = [1 / (i+1) for i in range(self.k)]
                weighted_sum = sum(neighbor[1] * weight for neighbor, weight in zip(nearest_neighbors, rank_weights))
                total_weight = sum(rank_weights)
                prediction = weighted_sum / total_weight

            predictions.append(prediction)

        return np.array(predictions)
