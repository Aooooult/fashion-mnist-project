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

def plot2DTrainTestSet(d_train,l_train, d_test,l_test, nom_dataset= "Dataset", avec_grid=True):    
    """ array * array array * array * str * bool-> affichage
        nom_dataset (str): nom du dataset pour la légende
        avec_grid (bool) : True si on veut afficher la grille, False sinon
        la fonction doit utiliser les couleurs suivantes:
        - pour les données d'apprentissage : la couleur 'red' pour la classe -1 et 'blue' pour la +1
        - pour les données de test : la couleur 'jaune' pour la classe -1 et 'verte' pour la +1
    """


    desc_train = np.asarray(d_train)
    desc_test = np.asarray(d_test)
    labels_train = np.asarray(l_train)
    labels_test = np.asarray(l_test)
    
    idx_neg_test = (l_test == -1)
    idx_pos_test = (l_test == 1)
    desc_neg_test = d_test[idx_neg_test]
    desc_pos_test = d_test[idx_pos_test]

    idx_neg_train = (l_train == -1)
    idx_pos_train = (l_train == 1)
    desc_neg_train = d_train[idx_neg_train]
    desc_pos_train = d_train[idx_pos_train]
    plt.scatter(desc_neg_test[:,0], desc_neg_test[:,1], marker='o', color="yellow", label="classe -1 test")
    plt.scatter(desc_pos_test[:,0], desc_pos_test[:,1], marker='x', color="green", label="classe +1 test") 
    plt.scatter(desc_neg_train[:,0], desc_neg_train[:,1], marker='o', color="red", label="classe -1")
    plt.scatter(desc_pos_train[:,0], desc_pos_train[:,1], marker='x', color="blue", label="classe +1") 
    plt.title(nom_dataset)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(avec_grid)
    plt.show()

