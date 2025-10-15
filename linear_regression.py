import numpy as np
import pandas as pd
import random

class MyLineReg:
    def __init__(self, n_iter=100, learning_rate=0.1, weights=None, metric=None, best_score=None, 
                 reg=None, l1_coef=0.0, l2_coef=0.0, sgd_sample=None, random_state=42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.weights = weights

        if isinstance(sgd_sample, float) and 0.0 < sgd_sample < 1.0:
            self.sgd_sample = sgd_sample
        elif isinstance(sgd_sample, int) and sgd_sample > 0:
            self.sgd_sample = sgd_sample
        else:
            self.sgd_sample = None

        self.metric = metric
        self.best_score = best_score

        self.reg = reg          
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

    def mean_squared_error(self, predictions: np.ndarray, y: pd.Series):
        mse = np.mean((predictions - y) ** 2)
        return mse

    def mean_absolute_error(self, predictions: np.ndarray, y: pd.Series):
        mae = np.mean(np.abs(predictions - y))
        return mae
    
    def root_mean_squared_error(self, predictions: np.ndarray, y: pd.Series):
        rmse = np.sqrt(np.mean((predictions - y) ** 2))
        return rmse
    
    def r2(self, predictions: np.ndarray, y: pd.Series):
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - predictions) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        return r2
    
    def mean_absolute_percentage_error(self, predictions: np.ndarray, y: pd.Series):
        mape = 100 * np.mean(np.abs((y - predictions) / y))
        return mape
    
    def lasso(self, l1_coef: float, predictions: np.ndarray, y: pd.Series):
        res = self.mean_squared_error(predictions, y) + l1_coef * np.sum(np.abs(self.weights))
        return res
    
    def ridge(self, l2_coef: float, predictions: np.ndarray, y: pd.Series):
        res = self.mean_squared_error(predictions, y) + l2_coef * np.sum(np.square(self.weights))
        return res
    
    def elasticnet(self, l1_coef: float, l2_coef: float, predictions: np.ndarray, y: pd.Series):
        res = self.mean_squared_error(predictions, y) + l1_coef * np.sum(np.abs(self.weights)) + l2_coef * np.sum(np.square(self.weights))
        return res

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False):
        bias_column = np.ones((X.shape[0], 1))
        X_with_bias = np.concatenate((bias_column, X), axis=1)
        
        num_features = X_with_bias.shape[1]

        if self.weights is None:
            self.weights = np.ones(num_features)

        random.seed(self.random_state)
        
        for i in range(1, self.n_iter + 1):
            if self.sgd_sample:
                if isinstance(self.sgd_sample, float):
                    sample_size = int(self.sgd_sample * X.shape[0])
                else:
                    sample_size = self.sgd_sample
                sample_rows_idx = random.sample(range(X.shape[0]), sample_size)
                X_sample = X_with_bias[sample_rows_idx]
                y_sample = y.iloc[sample_rows_idx]
            else:
                X_sample = X_with_bias
                y_sample = y

            predictions = np.dot(X_sample, self.weights)
            gradient = np.dot(X_sample.T, predictions - y_sample) * 2 / len(y_sample)
            
            if self.reg == 'l1':
                gradient += self.l1_coef * np.sign(self.weights)

            elif self.reg == 'l2':
                gradient += 2 * self.l2_coef * self.weights

            elif self.reg == 'elasticnet':
                gradient += self.l1_coef * np.sign(self.weights) + 2 * self.l2_coef * self.weights
            
            current_learning_rate = self.learning_rate(i) if callable(self.learning_rate) else self.learning_rate
            self.weights -= current_learning_rate * gradient
            
            if self.metric:
                score = self.calculate_metric(predictions, y_sample)
                if verbose and (i % 100 == 0 or i == self.n_iter):
                    print(f"{i} | loss: {self.mean_squared_error(predictions, y_sample):.2f} | {self.metric}: {score:.2f} | lr: {current_learning_rate:.6f}")
                self.best_score = score
        
        final_predictions = np.dot(X_with_bias, self.weights)
        self.best_score = self.calculate_metric(final_predictions, y)

    def predict(self, X: pd.DataFrame):
        bias_column = np.ones((X.shape[0], 1)) 
        X_with_bias = np.concatenate((bias_column, X), axis=1)
        return np.dot(X_with_bias, self.weights)

    def calculate_metric(self, predictions, y):
        if self.metric == 'mae':
            return self.mean_absolute_error(predictions, y)
        
        elif self.metric == 'mse':
            return self.mean_squared_error(predictions, y)
        
        elif self.metric == 'rmse':
            return self.root_mean_squared_error(predictions, y)
        
        elif self.metric == 'r2':
            return self.r2(predictions, y)
        
        elif self.metric == 'mape':
            return self.mean_absolute_percentage_error(predictions, y)
        
        else:
            return None

    def get_coef(self):
        return self.weights[1:]
    
    def get_best_score(self):
        return self.best_score
