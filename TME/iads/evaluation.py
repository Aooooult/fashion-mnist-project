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
from statistics import mean, stdev

# ------------------------ 


def crossval(X, Y, n_iterations, iteration):
    
    n = len(X)
    
    Xtest_id = list(range(iteration*n//n_iterations, (iteration+1)*n//n_iterations))
    
    Xapp_id = list(set(range(n)) - set(Xtest_id))
    Xtest = X[Xtest_id]
    Ytest = Y[Xtest_id]

    Xapp = X[Xapp_id]
    Yapp = Y[Xapp_id]
    
    return Xapp, Yapp, Xtest, Ytest




def crossval_strat(X, Y, n_iterations, iteration):
    # On récupère les classes présentes dans Y
    classes = np.unique(Y)
    
    Xtest_id = []
    Xapp_id = []
    
    # Pour chaque classe, on sélectionne les indices correspondants
    for c in classes:
        # Indices des éléments de la classe c
        indices_classe = np.where(Y == c)[0]
        n_classe = len(indices_classe)
        
        # Détermination des bornes pour l'itération courante
        debut = (iteration * n_classe) // n_iterations
        fin = ((iteration + 1) * n_classe) // n_iterations
        
        # Les indices de test sont ceux compris entre debut et fin
        Xtest_id.extend(indices_classe[debut:fin])
        
        # Les indices d'apprentissage sont ceux avant debut et après fin
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


def analyse_perfs(L):
    """ L : liste de nombres réels non vide
        rend le tuple (moyenne, écart-type)
    """
    return (float(np.mean(L)), float(np.std(L)))

def validation_croisee(C, DS, nb_iter, verbose = False):
    """ Classifieur * tuple[array, array] * int -> tuple[ list[float], float, float]
        Arguments:
            - C (Classifieur): un classifieur déjà défini (mais pas entraîné) 
            - DS (tuple[array,array]: un tuple composé d'un dataset (data, labels)
            - nb_iter (int): nombre d'itérations à réaliser
            - verbose: pour dire si on veut afficher des messages au cours de l'exécution
        Retour: tuple[ list[float], float, float]
            - triplet contenant la liste des performances obtenues, la performance moyenne et l'écart type
    """
    perf = list()    
    for i in range(nb_iter):
        Xapp,Yapp,Xtest,Ytest = crossval(DS[0], DS[1], nb_iter, i)
        C_copy = copy.deepcopy(C)
        C_copy.train(Xapp, Yapp)
        acc = C_copy.accuracy(Xtest, Ytest)
        perf.append(acc)
        if verbose: 
            print(f"Itération {i}\t: taille base app={len(Xapp)}\t taille base test={len(Xtest)}\t accuracy={acc:0.4f}")
    
    return (perf, mean(perf), stdev(perf))