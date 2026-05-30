# -*- coding: utf-8 -*-

"""
Package: iads
File: evaluation.py
Année: LU3IN026 - semestre 2 - 2025-2026, Sorbonne Université
"""

# ---------------------------
# Fonctions d'évaluation de classifieurs

# import externe
import numpy as np
import pandas as pd
import copy
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------ 


def crossval(X, Y, n_iterations, iteration):
    """ array * array * int * int -> tuple[array, array, array, array]
        Arguments:
            - X (array): matrice des descriptions
            - Y (array): vecteur des labels correspondants
            - n_iterations (int): nombre total de blocs (folds) pour la validation croisée
            - iteration (int): indice de l'itération courante (de 0 à n_iterations-1)
        Retour: tuple[array, array, array, array]
            - un tuple (Xapp, Yapp, Xtest, Ytest) où:
                - Xapp, Yapp: les descriptions et les labels pour l'apprentissage
                - Xtest, Ytest: les descriptions et les labels pour le test
    """
    n = len(X)
    Xtest_id = list(range(iteration*n//n_iterations, (iteration+1)*n//n_iterations))
    Xapp_id = list(set(range(n)) - set(Xtest_id))
    Xtest = X[Xtest_id]
    Ytest = Y[Xtest_id]
    Xapp = X[Xapp_id]
    Yapp = Y[Xapp_id]
    
    return Xapp, Yapp, Xtest, Ytest

def crossval_strat(X, Y, n_iterations, iteration):
    """ array * array * int * int -> tuple[array, array, array, array]
        Arguments:
            - X (array): matrice des descriptions
            - Y (array): vecteur des labels correspondants
            - n_iterations (int): nombre total de blocs (folds) pour la validation croisée
            - iteration (int): indice de l'itération courante (de 0 à n_iterations-1)
        Retour: tuple[array, array, array, array]
            - un tuple (Xapp, Yapp, Xtest, Ytest) où la proportion de chaque classe 
              est préservée (stratification) :
                - Xapp, Yapp: les descriptions et les labels pour l'apprentissage
                - Xtest, Ytest: les descriptions et les labels pour le test
    """
    # On récupère les classes présentes dans Y
    classes = np.unique(Y)
    
    Xtest_id = []
    Xapp_id = []
    
    # Pour chaque classe, on sélectionne les indices correspondants
    for c in classes:
        # Indices des éléments de la classe c
        indices_classe = np.where(Y == c)[0]
        n_classe = len(indices_classe)
        
        # Détermination des bornes pour l'itération courante au sein de cette classe
        debut = (iteration * n_classe) // n_iterations
        fin = ((iteration + 1) * n_classe) // n_iterations
        
        # Les indices de test sont ceux compris entre debut et fin pour la classe c
        Xtest_id.extend(indices_classe[debut:fin])
        
        # Les indices d'apprentissage sont ceux en dehors de cet intervalle
        Xapp_id.extend(indices_classe[:debut])
        Xapp_id.extend(indices_classe[fin:])
        
    # Conversion en tableau numpy et mélange pour ne pas avoir les classes groupées
    Xtest_id = np.array(Xtest_id)
    Xapp_id = np.array(Xapp_id)
    
    np.random.shuffle(Xtest_id)
    np.random.shuffle(Xapp_id)
    
    # Récupération des données correspondantes aux indices
    Xtest = X[Xtest_id]
    Ytest = Y[Xtest_id]
    
    Xapp = X[Xapp_id]
    Yapp = Y[Xapp_id]
    
    return Xapp, Yapp, Xtest, Ytest

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


def analyse_perfs(L):
    """ L : liste de nombres réels non vide
        rend le tuple (moyenne, écart-type)
    """
    return (float(np.mean(L)), float(np.std(L)))

def validation_croisee(C, DS, nb_iter, stratified=True, verbose=False):
    """ Classifier * tuple[array, array] * int * bool * bool -> tuple[list[float], float, float]
        Arguments:
            - C (Classifier): un classifieur déjà défini (mais pas entraîné)
            - DS (tuple[array, array]): un dataset (descriptions, labels)
            - nb_iter (int): nombre d'itérations pour la validation croisée
            - stratified (bool): si True, utilise crossval_strat, sinon crossval
            - verbose (bool): si True, affiche les détails de chaque itération
        Retour: tuple[list[float], float, float]
            - un triplet contenant :
                - la liste des performances (accuracy) de chaque itération
                - la performance moyenne
                - l'écart-type des performances
    """
    perf = []    
    for i in range(nb_iter):
        if stratified:
            Xapp, Yapp, Xtest, Ytest = crossval_strat(DS[0], DS[1], nb_iter, i)
        else:
            Xapp, Yapp, Xtest, Ytest = crossval(DS[0], DS[1], nb_iter, i)
            
        C_copy = copy.deepcopy(C)
        C_copy.train(Xapp, Yapp)
        acc = C_copy.accuracy(Xtest, Ytest)
        perf.append(acc)
        
        if verbose: 
            print(f"Itération {i}: accuracy={acc:0.4f}")
    
    moyenne, ecart_type = analyse_perfs(perf)
    return perf, moyenne, ecart_type

def matrice_de_confusion(C, DS, nb_iter, stratified=True, verbose=False): 
    labels = np.unique(DS[1])
    print(labels)
    for i in range(nb_iter):
        if stratified:
            Xapp, Yapp, Xtest, Ytest = crossval_strat(DS[0], DS[1], nb_iter, i)
        else:
            Xapp, Yapp, Xtest, Ytest = crossval(DS[0], DS[1], nb_iter, i)
            
        C_copy = copy.deepcopy(C)
        C_copy.train(Xapp, Yapp)
        y_pred = np.array([C_copy.predict(x) for x in Xtest])
    
        confucius = confusion_matrix(Ytest, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(confucius, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel('Prédictions')
        plt.ylabel('Vrai labels')
        plt.title('Matrice de Confusion')
        plt.show()