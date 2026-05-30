import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod
from sklearn.metrics import confusion_matrix

class Classifier(ABC):
    __nombre_crees: int = 0
    
    def __init__(self, input_dimension):
        Classifier.__nombre_crees += 1
        self.__ident = Classifier.__nombre_crees
        self.__dimension = input_dimension
        
    def get_dimension(self):
        return self.__dimension
    
    def get_w(self):
        return self.__w

    def set_w(self, new_w):
        self.__w = new_w
        
    def __str__(self) -> str:
        return f'Classifier #{self.__ident} (d{self.__dimension})'
        
    @abstractmethod
    def train(self, desc_set, label_set) -> None:
        pass

    @abstractmethod
    def score(self, x) -> float:
        pass
    
    @abstractmethod
    def predict(self, x) -> int:
        pass

    def accuracy(self, desc_set, label_set) -> float:
        sum_true = 0
        size_data = len(desc_set)
        for i, ligne in enumerate(desc_set): 
            if self.predict(ligne) == label_set[i]: 
                sum_true += 1
        return sum_true / size_data


class ClassifierPerceptron(Classifier):
    def __init__(self, input_dimension, learning_rate=0.01, init=True, verbose=False):
        super().__init__(input_dimension)
        self.__learning = learning_rate
        if init:
            self.__w = np.zeros(input_dimension)
        else:
            self.__w = (2 * np.random.rand(input_dimension) - 1) * 0.001
        if verbose:
            print(f"{super().__str__()}: initialisation (learning rate= {self.__learning}) w= {self.__w}")
        
    def train_step(self, desc_set, label_set):
        rng = np.random.default_rng()
        idx = list(range(len(label_set)))
        rng.shuffle(idx)
        for i in idx:
            x_i = desc_set[i]
            y_i = label_set[i]
            y = self.score(x_i)
            if (y_i * y) <= 0: 
               self.__w = self.__w + self.__learning * y_i * x_i  
     
    def __str__(self) -> str:
        return f'ClassifierPerceptron (d{self.get_dimension()})'

    def train(self, desc_set, label_set, nb_max=100, seuil=0.001, verbose=False):
        result_list = list()       
        for i in range(nb_max):
            w_old = self.__w.copy()
            self.train_step(desc_set, label_set)
            w_new = self.__w
            dif = np.linalg.norm(np.abs(w_new - w_old))
            result_list.append(dif)
            if dif < seuil:
                return result_list
        return result_list    
            
    def score(self, x):
        return np.dot(self.__w, x)
    
    def predict(self, x):
        return 1 if (self.score(x)) >= 0 else -1


class ClassifierKNN(Classifier):
    def __init__(self, input_dimension, k):
        super().__init__(input_dimension)
        self.__k = k
        self.__desc_set = None   
        self.__labels_set = None

    def __str__(self) -> str:
        return f"ClassifierKNN k={self.__k} (d{self.get_dimension()})"

    def train(self, desc_set, label_set) -> None:
        self.__desc_set = desc_set
        self.__labels_set = label_set

    def score(self, x) -> float:
        distances = np.linalg.norm((self.__desc_set - x), axis=1)
        k_near_id = np.argsort(distances)[:self.__k]
        labels = self.__labels_set[k_near_id]
        p = np.sum((labels + 1) / 2) / self.__k
        return 2 * (p - 0.5)

    def predict(self, x) -> int:
        if (self.score(x) > 0):
            return 1
        return -1


class ClassifierMultiOAA(Classifier):
    def __init__(self, input_dimension, cl_bin):
        super().__init__(input_dimension)
        self.cl_bin = cl_bin
        self.classifiers = []
        self.classes = []       
        
    def train(self, desc_set, label_set):
        self.classes = np.unique(label_set)
        self.classifiers = []
        for c in self.classes:
            classifier_c = copy.deepcopy(self.cl_bin)
            y_tmp = np.where(label_set == c, 1, -1)
            classifier_c.train(desc_set, y_tmp)
            self.classifiers.append(classifier_c)
    
    def score(self, x):
        return [classifier.score(x) for classifier in self.classifiers]
        
    def predict(self, x):
        scores = self.score(x)
        idx_max = np.argmax(scores)
        return self.classes[idx_max]


class ClassifierPerceptronStable(ClassifierPerceptron):
    def __init__(self, input_dimension, learning_rate=0.01, init=True, verbose=False):
        super().__init__(input_dimension, learning_rate, init, verbose)
        if verbose:
            print(f"{super().__str__()} (stabilise)")

    def __str__(self) -> str:
        return f"{super().__str__()} (stabilise)"
    
    def train(self, desc_set, label_set, nb_max=100, seuil=0.001, stabilised=True, verbose=False):
        result_list = list()
        
        if not stabilised:
            return super().train(desc_set, label_set, nb_max, seuil, verbose)
            
        best_w = self.get_w().copy()
        best_accuracy = self.accuracy(desc_set, label_set)
        
        for i in range(nb_max):
            w_old = self.get_w().copy()
            
            self.train_step(desc_set, label_set)
            
            w_new = self.get_w()
            dif = np.linalg.norm(np.abs(w_new - w_old))
            result_list.append(dif)
            
            current_accuracy = self.accuracy(desc_set, label_set)
            
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_w = w_new.copy()
                if verbose:
                    print(f"Etape {i}: Nouvelle meilleure accuracy = {best_accuracy:.4f}")
            
            if dif < seuil:
                break
                
        self.set_w(best_w)
        
        return result_list


class ClassifierPerceptronKernel(ClassifierPerceptron):
    def __init__(self, input_dimension, learning_rate, noyau, init=True, verbose=False):
        super().__init__(noyau.get_output_dim(), learning_rate, init)
        self.noyau = noyau
        if verbose:
            print(f"{super().__str__()}: initialisation avec kernalisation")
        
    def train_step(self, desc_set, label_set, stabilised=False):
        desc_set_transformed = self.noyau.transform(desc_set)
        super().train_step(desc_set_transformed, label_set)     
     
    def score(self, x):
        if len(x) == self.noyau.get_input_dim():
            x = self.noyau.transform(x.reshape(1, -1))[0]
        return super().score(x)