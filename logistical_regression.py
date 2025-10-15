import numpy as np
import pandas as pd
import random

class MyLogReg:
    def __init__(self, n_iter=10, learning_rate=0.1, weights=None, metric=None, 
                 best_score=None, reg=None, l1_coef=0.0, l2_coef=0.0, sgd_sample=None, random_state=42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate

        self.weights = weights

        self.metric = metric
        self.best_score = best_score

        self.reg = reg          
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def lassoLogLoss(self, X, pred_y, y, l1_coef):
        lasso = np.mean((np.dot((pred_y - y), X))) + l1_coef * np.sign(self.weights)
        return lasso
    
    def ridgeLogLoss(self, X, pred_y, y, l2_coef):
        ridge = np.mean((np.dot((pred_y - y), X))) + l2_coef * 2 * self.weights
        return ridge

    def elasticNetLogLoss(self, X, pred_y, y, l1_coef, l2_coef):
        elasticnet = np.mean((np.dot((pred_y - y), X))) + l1_coef * np.sign(self.weights) + l2_coef * 2 * self.weights
        return elasticnet

    def accuracy(self, tp, tn, fp, fn):
        return (tp + tn) / (tp + tn + fp + fn)
    
    def precision(self, tp, fp):
        return tp / (tp + fp)
    
    def recall(self, tp, fn):
        return tp / (tp + fn)
    
    def f1(self, tp, fp, fn, beta=1):
        precision_value = self.precision(tp, fp)
        recall_value = self.recall(tp, fn)
        return (1 + beta**2) * ((precision_value * recall_value) / (beta**2 * precision_value + recall_value))
    
    def roc_auc(self, y_true, y_pred_proba):
        data = pd.DataFrame({'true_label': y_true, 'predicted_prob': y_pred_proba})
        data = data.sort_values('predicted_prob', ascending=False).reset_index(drop=True)
        
        positives = sum(data['true_label'])
        negatives = len(data) - positives
        
        tpr = np.zeros(len(data) + 1)
        fpr = np.zeros(len(data) + 1)
        
        for i in range(1, len(data) + 1):
            tpr[i] = tpr[i - 1]
            fpr[i] = fpr[i - 1]
            
            if data.loc[i - 1, 'true_label'] == 1:
                tpr[i] += 1 / positives
            else:
                fpr[i] += 1 / negatives
        
        auc = np.trapz(tpr, fpr)
        auc_rounded = round(auc, 10)
        return auc_rounded

    def log_loss(self, y, pred_y):
        eps = 1e-15
        return -np.mean(y * np.log(pred_y + eps) + (1 - y) * np.log(1 - pred_y + eps))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def log_loss_gradient(self, y, pred_y, X_with_bias):
        return X_with_bias.T @ (pred_y - y) / y.size
    
    def compute_confusion_matrix(self, y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp, tn, fp, fn

    def fit(self, X, y, verbose=False):
        bias_column = np.ones((X.shape[0], 1))
        X_with_bias = np.concatenate((bias_column, X), axis=1)
        
        num_features = X_with_bias.shape[1]

        if self.weights is None:
            self.weights = np.ones(num_features)

        random.seed(self.random_state)

        for i in range(1, self.n_iter + 1):
            lin_reg_ans = X_with_bias @ self.weights
            predict_proba = self.sigmoid(lin_reg_ans)
            
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

            lin_reg_ans_sample = X_sample @ self.weights
            predict_proba_sample = self.sigmoid(lin_reg_ans_sample)
            gradient = self.log_loss_gradient(y_sample, predict_proba_sample, X_sample)

            if self.reg == 'l1':
                gradient += self.l1_coef * np.sign(self.weights)

            elif self.reg == 'l2':
                gradient += 2 * self.l2_coef * self.weights

            elif self.reg == 'elasticnet':
                gradient += self.l1_coef * np.sign(self.weights) + 2 * self.l2_coef * self.weights

            if callable(self.learning_rate):
                clr_lambda = self.learning_rate(i)
                self.weights -= clr_lambda * gradient
            else:
                crl = self.learning_rate
                self.weights -= crl * gradient

        final_predict_proba = self.predict_proba(X)
        final_y_pred = (final_predict_proba > 0.5).astype(int)
        tp, tn, fp, fn = self.compute_confusion_matrix(y, final_y_pred)

        metric_value = self.calculate_metric(tp, tn, fp, fn, y, final_predict_proba)
        self.best_score = metric_value

    def calculate_metric(self, tp, tn, fp, fn, y_true=None, y_pred_proba=None):
        if self.metric == 'accuracy':
            return self.accuracy(tp, tn, fp, fn)
        
        elif self.metric == 'precision':
            return self.precision(tp, fp)
        
        elif self.metric == 'recall':
            return self.recall(tp, fn)
        
        elif self.metric == 'f1':
            return self.f1(tp, fp, fn)
        
        elif self.metric == 'roc_auc' and y_true is not None and y_pred_proba is not None:
            return self.roc_auc(y_true, y_pred_proba)
        
        else:
            return None

    def predict_proba(self, X):
        bias_column = np.ones((X.shape[0], 1))
        X_with_bias = np.concatenate((bias_column, X), axis=1)
        return self.sigmoid(X_with_bias @ self.weights)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)

    def get_coef(self):
        return self.weights[1:]
    
    def get_best_score(self):
        return self.best_score
