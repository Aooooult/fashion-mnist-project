# -*- coding: utf-8 -*-

"""
Package: iads
File: Classifiers.py
Année: LU3IN026 - semestre 2 - 2025-2026, Sorbonne Université
"""

# Classfieurs implémentés en LU3IN026
# Version de départ : Février 2026

# Import de packages externes
import numpy as np
import pandas as pd

import copy

from abc import ABC, abstractmethod

# ---------------------------

class Classifier(ABC):
    """ Classe (abstraite) pour représenter un classifieur
        Attention: cette classe est ne doit pas être instanciée.
    """
    __nombre_crees: int = 0  # Variable de classe pour compter le nombre de classifiers créés
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        Classifier.__nombre_crees += 1
        self.__ident = Classifier.__nombre_crees  # identifiant du classifieur (unique)
        self.__dimension = input_dimension
        
    def get_dimension(self):
        """ Accesseur de la variable __dimension 
        """
        return self.__dimension
        
    def __str__(self) -> str:
        """ rend une chaîne de caractères (méthode toString)
            Par exemple, pour afficher des informations sur l'objet 
        """
        return f'Classifier #{self.__ident} (d{self.__dimension})'
        
    @abstractmethod
    def train(self, desc_set, label_set) -> None:
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: array avec des descriptions
            label_set: array avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        pass

    @abstractmethod
    def score(self,x) -> float:
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        pass
    
    @abstractmethod
    def predict(self, x) -> int:
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        pass

    def accuracy(self, desc_set, label_set) -> float:
        """ rend le taux d'exemples bien classés dans le dataset
            desc_set: array avec des descriptions
            label_set: array avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        sum_true = 0
        size_data = len(desc_set)
        for i, ligne in enumerate(desc_set): 
            prediction = self.predict(ligne)  
            if prediction == label_set[i]: 
                sum_true += 1
        return sum_true / size_data


##### K-PP (TME2) #######

class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """
    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        super().__init__(input_dimension)  # Appel du constructeur de la classe mère
        self.__k= k
        # les 2 variables suivantes seront utilisées dans la méthode train()
        self.__desc_set= None   
        self.__labels_set= None

    def __str__(self) -> str:
        """ rend une chaîne de caractères (méthode toString)
            Par exemple, pour afficher des informations sur l'objet 
        """
        return f'ClassifierKNN (dim: {self.get_dimension()}, k: {self.__k})'
        #raise NotImplementedError("Please Implement this method")

    def train(self, desc_set, label_set) -> None:
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: array avec des descriptions
            label_set: array avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """   
        self.__desc_set = desc_set
        self.__labels_set = label_set
        #raise NotImplementedError("Please Implement this method")

    def score(self,x) -> float:
        """ rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un array
        """
        distances = np.linalg.norm(self.__desc_set - x, axis=1)
        indices_k_proches = np.argsort(distances)[:self.__k]
        voisins_labels = self.__labels_set[indices_k_proches]
        proportion_pos = np.sum(voisins_labels == 1) / self.__k
        return 2 * (proportion_pos - 0.5)
        #raise NotImplementedError("Please Implement this method")
    
    def predict(self, x) -> int:
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un array
        """
        return 1 if self.score(x) > 0 else -1
        #raise NotImplementedError("Please Implement this method")


class ClassifierKNN_MC(Classifier):
    def __init__(self, input_dimension, k, nc):
        super().__init__(input_dimension)
        self.k = k
        self.nc = nc
        self.desc_set = None
        self.label_set = None
    
    def __str__(self) -> str:
        return f'ClassifierKNN_MC (dim: {self.get_dimension()}, k: {self.k}, nc: {self.nc})'
    
    def train(self, desc_set, label_set) -> None:       
        self.desc_set = desc_set
        self.label_set = label_set
    
    def score(self, x) -> np.ndarray:
        distances = np.linalg.norm(self.desc_set - x, axis=1)
        indices_k_proches = np.argsort(distances)[:self.k]
        voisins_labels = self.label_set[indices_k_proches]
        votes = np.zeros(self.nc)
        for label in voisins_labels:
            votes[int(label)] += 1
            
        return votes
        
    def predict(self, x) -> int:
        votes = self.score(x)
        return np.argmax(votes)