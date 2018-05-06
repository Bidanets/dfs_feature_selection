import numpy as np
from sklearn import svm
from sklearn.linear_model import LinearRegression
import optunity.metrics as metrics
from sklearn.neural_network import MLPRegressor as mlpr
from pylab import rcParams
import pandas as pd
import data_preprocessor
import random
import copy
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML
import matplotlib
import matplotlib.animation as animation
import matplotlib.patches as patches
from pylab import rcParams

class Q_Criterion:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        
        self.X_test = X_test
        self.y_test = y_test
        
        self.n = len(X_train[0])
        
    def get_list_of_indexes_by_bitmask(self, bitmask):
        ls = []
        for i in range(len(bitmask)):
            if bitmask[i] == 1:
                ls.append(i)
                
        return ls 
    
    def calc_Q(self, list_of_feature_indexes):
        feature_bitmask = []
        for i in range(self.n):
            if i in list_of_feature_indexes:
                feature_bitmask.append(1)
            else:
                feature_bitmask.append(0)
            
        list_of_features = self.get_list_of_indexes_by_bitmask(feature_bitmask)
        
        model = LinearRegression()
        model.fit(self.X_train[:, list_of_features], self.y_train)
        
        y_hat = model.predict(self.X_test[:, list_of_features])
        Q = metrics.mse(y_hat, self.y_test)
        return Q
        
        
class DFS:
    INF = 2E9
    Q_best = INF
    F_best = []
    
    def __init__(self, n):
        self.n = n
        self.Q_min = [self.INF]*(n+1)
        return
            
    # -------------------------------------------------------------------------
    
    
    def draw_Q(self, Q):
        x = []
        y = []
        
        for i in range(len(Q)):
            if Q[i] != self.INF:
                x.append(i+1)
                y.append(Q[i])
                
        index_of_best = y.index(min(y))
                
        plt.plot(x, y, 'ro-', color = 'green', label = 'min Q')
        plt.plot(index_of_best+1, y[index_of_best], 'ro', color = 'red', label = 'best Q')
        
        print(index_of_best+1, y[index_of_best])

        plt.xlabel('Number of features')
        plt.ylabel('Q')
        plt.title('Q best for each set complexity')   
        
        plt.legend()
        
        plt.rcParams["figure.figsize"] = (12, 4)
        
        plt.show()
    
    # ------------------------------------------------------------------------- 
    
    def build_up(self, F):
        cardinality = len(F)
        
        if len(F) == 0:
            Q_F = self.INF
        else:
            Q_F = self.q_criterion.calc_Q(F)
        
        if self.Q_best == Q_F and len(F) < len(self.F_best):
            self.F_best = copy.deepcopy(F)
        
        if self.Q_best > Q_F:
            self.Q_best = copy.deepcopy(Q_F)
            self.F_best = copy.deepcopy(F)
            
        print('F: ', F)
        print('Q: ', Q_F)
        print(30 * '-')
        
        for j in range(1, cardinality - self.d + 1):
            if Q_F >= self.kapa * self.Q_min[j]:
                return
        
        self.Q_min[cardinality] = min(self.Q_min[cardinality], Q_F)
        
        
        for s in self.indexes_of_features:
            max_t = -1
            if len(F) > 0:
                max_t = max(F)
            if s > max_t:
                self.build_up(F + [s])
                
        return 
        
        
        

    
    def fit(self, X_train, y_train, q_criterion, d, kapa):
        # Initializing an Array of Best Criteria Values
        
        self.d = d
        self.kapa = kapa
        self.q_criterion = q_criterion
        
        self.n = len(X_train[0])
        
        """
        # heuristic: сортировка признаков по информативности
        Q_min = []
        for j in range(self.n):
            Q_min.append(self.INF)
            
        
        Q = []
            
        cur_bitmask = [0] * self.n
        for j in range(self.n):
            cur_bitmask[j] = 1
            Q.append(q_criterion.calc_Q(cur_bitmask))
            cur_bitmask[j] = 0
        
        indexes_of_features = [x for _,x in sorted(zip(Q, indexes_of_features))]
        """
        
        self.indexes_of_features = list(np.arange(0, self.n-1))
        self.build_up([])
        
        return 
    
    def get_best_F(self):
        F = []
        for i in range(self.n):
            if i in self.F_best:
                F.append(1)
            else:
                F.append(0)
                
        return F
                
        
    # -------------------------------------------------------------------------
            
def print_feature_names(F, names_of_features):
    print('Features:', end = ' ')
    for i in range(len(F)):
        if F[i] == 1:
            print(names_of_features[i], end = ' ')
    print()
    
        
def main():
    dp = data_preprocessor.Data_Preprocessor()
    X_train, y_train, X_test, y_test, names_of_features = dp.Prepare_data('boston')
    
    q_criterion = Q_Criterion(X_train, y_train, X_test, y_test)
    
    dfs = DFS(n = len(X_train[0]))
    dfs.fit(X_train, y_train, q_criterion, d = 3, kapa = 1.1)    
    
    F = dfs.get_best_F()
    
    dfs.draw_Q(dfs.Q_min[1:])
    
    print_feature_names(F, names_of_features)
    
    
main() 
        