import numpy as np
import pandas as pd

class MyKNNReg:
    def __init__(self, k=3, metric='euclidean'):
        self.k = k
        self.X = None
        self.y = None
        self.train_size = None
        self.metric = metric

    def euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def chebyshev_distance(self, point1, point2):
        return np.max(np.abs(point1 - point2))

    def manhattan_distance(self, point1, point2):
        return np.sum(np.abs(point1 - point2))

    def cosine_distance(self, point1, point2):
        return 1 - np.dot(point1, point2) / (np.linalg.norm(point1) * np.linalg.norm(point2))
    
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
            classes = [neighbor[1] for neighbor in nearest_neighbors]
            prediction = max(set(classes), key=classes.count)

            if classes.count(0) == classes.count(1):
                prediction = 1
            predictions.append(prediction)

        return np.array(predictions)
