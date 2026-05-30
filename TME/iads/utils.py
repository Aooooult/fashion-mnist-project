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

# genere_dataset_uniform:
def genere_dataset_uniform(d, nc, binf=-1, bsup=1):
    """ int * int * float^2 -> tuple[array, array]
        Hyp: n est pair
        d: nombre de dimensions de la description
        nc: nombre d'exemples de chaque classe
        les valeurs générées uniformément sont dans [binf,bsup]
    """
    tuple_1 = np.random.uniform(binf, bsup, (2*nc, d))
    tuple_2 = np.array([-1 for i in range(0,nc)] + [+1 for i in range(0,nc)])
    return (tuple_1, tuple_2)


# genere_dataset_gaussian:
def genere_dataset_gaussian(positive_center, positive_sigma, negative_center, negative_sigma, nc):
    """ les valeurs générées suivent une loi normale
        rend un tuple (data_desc, data_labels)
    """
    pos = np.random.multivariate_normal(positive_center, positive_sigma, nc)
    neg = np.random.multivariate_normal(negative_center, negative_sigma, nc)
    data_desc = np.vstack((neg, pos))
    data_label = np.array([-1 for i in range(0,nc)] + [+1 for i in range(0,nc)])
    return (data_desc, data_label)

# plot2DSet:
def plot2DSet(desc, labels, nom_dataset="Dataset", avec_grid=True):    
    desc = np.asarray(desc)
    labels = np.asarray(labels)
    idx_neg = (labels == -1)
    idx_pos = (labels == 1)
    desc_neg = desc[idx_neg]
    desc_pos = desc[idx_pos]
    plt.scatter(desc_neg[:,0], desc_neg[:,1], marker='o', color="red", label="classe -1")
    plt.scatter(desc_pos[:,0], desc_pos[:,1], marker='x', color="blue", label="classe +1") 
    plt.title(nom_dataset)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(avec_grid)
    plt.show()

# plot_frontiere:
def plot_frontiere(desc_set, label_set, classifier, step=30):
    """ desc_set * label_set * Classifier * int -> NoneType
        Remarque: le 4e argument est optionnel et donne la "résolution" du tracé: plus il est important
        et plus le tracé de la frontière sera précis.        
        Cette fonction affiche la frontière de décision associée au classifieur
    """
    mmax=desc_set.max(0)
    mmin=desc_set.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    
    # calcul de la prediction pour chaque point de la grille
    res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    res=res.reshape(x1grid.shape)
    # tracer des frontieres
    # colors[0] est la couleur des -1 et colors[1] est la couleur des +1
    plt.contourf(x1grid,x2grid,res,colors=["darksalmon","skyblue"],levels=[-1000,0,1000])


def genere_train_test(desc_set, label_set, n_pos, n_neg):
    """ permet de générer une base d'apprentissage et une base de test
        desc_set: array avec des descriptions
        label_set: array avec les labels correspondants
        n_pos: nombre d'exemples de label +1 à mettre dans la base d'apprentissage
        n_neg: nombre d'exemples de label -1 à mettre dans la base d'apprentissage
        Hypothèses: 
           - desc_set et label_set ont le même nombre de lignes)
           - n_pos et n_neg, ainsi que leur somme, sont inférieurs à n (le nombre d'exemples dans desc_set)
    """
    # 1. Séparation des indices par classe
    indices_pos = np.where(label_set == 1)[0]
    indices_neg = np.where(label_set == -1)[0]
    
    # 2. Mélange aléatoire des indices de chaque classe
    rng = np.random.default_rng()
    rng.shuffle(indices_pos)
    rng.shuffle(indices_neg)
    
    # 3. Sélection des indices pour le TRAIN
    train_idx_pos = indices_pos[:n_pos]
    train_idx_neg = indices_neg[:n_neg]
    
    # 4. Sélection des indices pour le TEST (le reste des données)
    test_idx_pos = indices_pos[n_pos:]
    test_idx_neg = indices_neg[n_neg:]
    
    # 5. Regroupement des indices (Train ensemble et Test ensemble)
    train_indices = np.concatenate([train_idx_pos, train_idx_neg])
    test_indices = np.concatenate([test_idx_pos, test_idx_neg])
    
    # Optionnel: on peut mélanger les indices finaux pour ne pas avoir 
    # tous les +1 suivis de tous les -1
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    
    # 6. Création des bases finales en utilisant le slicing par indices
    train_desc = desc_set[train_indices]
    train_label = label_set[train_indices]
    
    test_desc = desc_set[test_indices]
    test_label = label_set[test_indices]
    
    return ((train_desc, train_label), (test_desc, test_label))

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

def create_XOR(n, var):
    """ int * float -> tuple[ndarray, ndarray]
        Hyp: n et var sont positifs
        n: nombre de points voulus
        var: variance sur chaque dimension
    """
    sigma = np.sqrt(var)
    n_par_nuag = n // 4

    gaus = np.random.normal(0, sigma, (n, 2))
    
    #nuage avec coord (1,1)
    gaus[0:n_par_nuag, 0] += 1
    gaus[0:n_par_nuag, 1] += 1
    
    #(-1, -1)
    gaus[n_par_nuag:2*n_par_nuag, 0] -= 1
    gaus[n_par_nuag:2*n_par_nuag, 1] -= 1
    
    # (1, -1)
    gaus[2*n_par_nuag:3*n_par_nuag, 0] += 1
    gaus[2*n_par_nuag:3*n_par_nuag, 1] -= 1
    
    #Центр (-1, 1)
    gaus[3*n_par_nuag:, 0] -= 1
    gaus[3*n_par_nuag:, 1] += 1
    

    labels = np.ones(n)
    labels[2*n_par_nuag:] = -1
    
    return gaus, labels