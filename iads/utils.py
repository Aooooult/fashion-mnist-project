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

