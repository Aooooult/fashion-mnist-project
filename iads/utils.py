# -*- coding: utf-8 -*-

"""
Package: iads
File: utils.py
Année: LU3IN026 - semestre 2 - 2025-2026, Sorbonne Université
"""


# Fonctions utiles
# Version de départ : Février 2026

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------ 
def echantillonnage_homogene(X, Y, proportion):
    """ array * array * float -> tuple[array, array]
        Arguments:
            - X (array): matrice des descriptions
            - Y (array): vecteur des labels correspondants
            - proportion (float): la fraction de données à extraire (entre 0 et 1)
        Retour: tuple[array, array]
            - un tuple (X_sub, Y_sub) contenant un sous-ensemble des données
              préservant les proportions de chaque classe.
    """
    # On récupère les classes présentes dans Y
    classes = np.unique(Y)
    
    indices_selectionnes = []
    
    for c in classes:
        # 1. On récupère tous les indices de la classe c
        indices_classe = np.where(Y == c)[0]
        
        # 2. On calcule combien d'éléments on doit prendre pour cette classe
        # (proportion * nombre total d'éléments de la classe)
        n_a_prendre = int(len(indices_classe) * proportion)
        
        # 3. On mélange les indices de cette classe pour avoir un tirage aléatoire
        indices_melanges = np.random.permutation(indices_classe)
        
        # 4. On sélectionne les n premiers indices mélangés
        indices_selectionnes.extend(indices_melanges[:n_a_prendre])
        
    # On convertit en tableau numpy et on remélange le tout pour ne pas avoir les classes groupées
    indices_selectionnes = np.array(indices_selectionnes)
    np.random.shuffle(indices_selectionnes)
    
    # Récupération des données correspondantes
    X_sub = X[indices_selectionnes]
    Y_sub = Y[indices_selectionnes]
    
    return X_sub, Y_sub

def normalisation(df):
    df_norm = df.copy()
    for col in df_norm.columns:
        mini = df_norm[col].min()
        maxi = df_norm[col].max()
        # Normalisation min-max dans [0,1]
        df_norm[col] = (df_norm[col] - mini) / (maxi - mini)
    return df_norm

from abc import ABC, abstractmethod

class Distance(ABC):
    """ Classe abstraite pour représenter des mesures de distances
        Elle permet de définir une hiérarchie pour les distances
    """
    def __init__(self,nom):
        """ Constructeur:
            prend en argument le nom (str) de la distance créé
        """
        self.__nom:str = nom

    @abstractmethod
    def calcule(self, v, M):
        """ Arguments:
                - v: un vecteur
                - M: un vecteur ou une matrice
            Hypothèse: v et M ont le même nombre de colonnes
            Retour:
                - un float si M est une vecteur: distance entre v et M
                - une np.series si M est une matrice: distances entre le vecteur v et chaque vecteur de M
        """
        # le calcul de distance dépend de la mesure que l'on utilise, il sera implémenté
        # dans les sous-classes de cette classe.
        pass

    def __str__(self) -> str:
        """ rend une chaîne de caractères (méthode toString)
            Par exemple, pour afficher des informations sur l'objet
        """
        return "Distance "+self.__nom

class DistanceEuclidienne(Distance):
    """ Classe représentant la distance euclidienne
    """
    def __init__(self):
        """ Constructeur
        """
        super().__init__("euclidienne")

    def calcule(self, v, M):
        """ Arguments:
                - v: un vecteur
                - M: un vecteur ou une matrice
            Hypothèse: v et M ont le même nombre de colonnes
            Retour:
                - un float si M est une vecteur: distance entre v et M
                - une np.series si M est une matrice: distances entre le vecteur v et chaque vecteur de M
        """
        if v.ndim != 1:
            raise TypeError("Argument incorrect: le premier argument doit être un vecteur")

        # Cas vecteur
        if M.ndim == 1:
            return np.linalg.norm(M - v)

        # Cas matrice
        elif M.ndim == 2:
            return np.linalg.norm(M - v, axis=1)

        else:
            raise TypeError("Argument incorrect: M doit être un vecteur ou une matrice")


    def __str__(self) -> str:
        """ rend une chaîne de caractères (méthode toString)
            Par exemple, pour afficher des informations sur l'objet
        """
        return super().__str__()

class DistanceMinkowski(Distance):
    """ Classe de distance Minkowski
    """
    def __init__(self, p=3):
        """ Constructeur
            Argument:
                - p: ordre de la distance de Minkowski (p>=1)
        """
        if p < 1:
            raise ValueError("L'ordre p doit être supérieur ou égal à 1")
        super().__init__(f"Minkowski(p={p})")
        self.__p = p

    def calcule(self, v, M):
        """ Arguments:
                - v: un vecteur
                - M: un vecteur ou une matrice
            Hypothèse: v et M ont le même nombre de colonnes
            Retour:
                - un float si M est une vecteur: distance entre v et M
                - une np.series si M est une matrice: distances entre le vecteur v et chaque vecteur de M
        """
        if v.ndim != 1:
            raise TypeError("Argument incorrect: le premier argument doit être un vecteur")

        # Cas vecteur
        if M.ndim == 1:
            return np.sum(np.abs(M - v) ** self.__p) ** (1 / self.__p)

        # Cas matrice
        elif M.ndim == 2:
            return np.sum(np.abs(M - v) ** self.__p, axis=1) ** (1 / self.__p)

        else:
            raise TypeError("Argument incorrect: M doit être un vecteur ou une matrice")