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
import scipy.cluster.hierarchy

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

def CHA_initialise(DF):
    """ Initialise la partition de départ pour CHA
        Argument:
            - DF (DataFrame): base d'apprentissage
        Retour:
            - dict: clé = numéro de cluster (entier), valeur = liste des indices d'exemples
    """
    partition = {}
    for i, idx in enumerate(DF.index):
        partition[i] = [idx]
    return partition

def CHA_fusionne(df, p0, linkage_instance, verbose=False):
    """ Fusionne les 2 groupes les plus proches selon le linkage
        Arguments:
            - df: DataFrame de la base d'apprentissage
            - p0: partition
            - linkage_instance: instance de la classe Linkage à utiliser pour calculer les distances entre groupes
            - verbose: affichage du debuggage
        Retour:
            - (p1, id0, id1, dist): p1: nouvelle partition après fusion, id0 et id1: indices des groupes fusionnés, dist: distance entre les groupes fusionnés
    """

    ids = sorted(p0.keys())
    best_id0, best_id1 = None, None
    best_dist = np.inf

    # OPTIMISATION MAJEURE : On pré-extrait les groupes de Pandas vers Numpy 
    # une seule fois au lieu de le faire dans la double boucle pour chaque paire.
    # Cela évite des millions d'appels très lents à df.loc.
    groupes = {id_c: df.loc[p0[id_c]].values for id_c in ids}

    # Recherche des 2 clusters les plus proches
    for i in range(len(ids)):
        id0 = ids[i]
        G0 = groupes[id0]
        for j in range(i + 1, len(ids)):
            id1 = ids[j]
            G1 = groupes[id1]
            dist = linkage_instance.calcule(G0, G1, verbose=False)

            if dist < best_dist:
                best_dist = dist
                best_id0, best_id1 = id0, id1

    # Construction de la nouvelle partition
    p1 = {k: list(v) for k, v in p0.items() if k not in (best_id0, best_id1)}
    new_id = max(p0.keys()) + 1
    p1[new_id] = p0[best_id0] + p0[best_id1]

    if verbose:
        print(f"Fusion de {best_id0} et {best_id1} --> distance = {best_dist:1.4f}")

    return p1, best_id0, best_id1, best_dist

def CHA_algorithme(df, linkage_instance, verbose=False):
    """ Algorithme de CHA
        Arguments:
            - df: DataFrame de la base d'apprentissage
            - linkage_instance: instance de la classe Linkage à utiliser pour calculer les distances entre groupes
            - verbose: affichage du debuggage
        Retour:
            - liste resultat: [[id0, id1, dist, taille], ...]
    """
    p = CHA_initialise(df)
    resultat = []
    
    nb_initial = len(p)
    if verbose:
        print(f"Début de CHA_algorithme sur {nb_initial} exemples.")

    while len(p) > 1:
        if verbose:
            print(f"[{nb_initial - len(p) + 1}/{nb_initial - 1}] Clusters restants : {len(p)}...", end=' ')

        # La taille du cluster fusionne se calcule sur la partition avant fusion
        p1, id0, id1, dist = CHA_fusionne(df, p, linkage_instance, verbose)
        taille = len(p[id0]) + len(p[id1])
        resultat.append([id0, id1, float(dist), taille])
        p = p1

    if verbose:
        print("Algorithme CHA terminé.")

    return resultat

def CHA_dendrogramme(results, linkage_info):
    """ Affiche le dendrogramme à partir des résultats de CHA
        Arguments:
            - results: liste des fusions effectuées par CHA (id0, id1, dist, taille)
            - linkage_info: str, informations sur le linkage utilisé (pour le titre du graphique)
    """

    # Paramètre de la fenêtre d'affichage:
    plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
    plt.title(f'Dendrogramme - {linkage_info}', fontsize=25)
    plt.xlabel("Indice d'exemple", fontsize=25)
    plt.ylabel('Distance', fontsize=25)

    # Construction du dendrogramme pour notre clustering :
    scipy.cluster.hierarchy.dendrogram(
        results,
        leaf_font_size=24.,  # taille des caractères de l'axe des X
    )

    # Affichage du résultat obtenu:
    plt.show()