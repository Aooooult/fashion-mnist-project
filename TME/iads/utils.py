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


# genere_dataset_uniform:
def genere_dataset_uniform(d, nc, binf=-1, bsup=1):
    """int * int * float^2 -> tuple[array, array]
    Hyp: n est pair
    d: nombre de dimensions de la description
    nc: nombre d'exemples de chaque classe
    les valeurs générées uniformément sont dans [binf,bsup]
    """
    tuple_1 = np.random.uniform(binf, bsup, (2 * nc, d))
    tuple_2 = np.array([-1 for i in range(0, nc)] + [+1 for i in range(0, nc)])
    return (tuple_1, tuple_2)


# genere_dataset_gaussian:
def genere_dataset_gaussian(
    positive_center, positive_sigma, negative_center, negative_sigma, nc
):
    """les valeurs générées suivent une loi normale
    rend un tuple (data_desc, data_labels)
    """
    pos = np.random.multivariate_normal(positive_center, positive_sigma, nc)
    neg = np.random.multivariate_normal(negative_center, negative_sigma, nc)
    data_desc = np.vstack((neg, pos))
    data_label = np.array([-1 for i in range(0, nc)] + [+1 for i in range(0, nc)])
    return (data_desc, data_label)


# plot2DSet:
def plot2DSet(desc, labels, nom_dataset="Dataset", avec_grid=True):
    desc = np.asarray(desc)
    labels = np.asarray(labels)
    idx_neg = labels == -1
    idx_pos = labels == 1
    desc_neg = desc[idx_neg]
    desc_pos = desc[idx_pos]
    plt.scatter(
        desc_neg[:, 0], desc_neg[:, 1], marker="o", color="red", label="classe -1"
    )
    plt.scatter(
        desc_pos[:, 0], desc_pos[:, 1], marker="x", color="blue", label="classe +1"
    )
    plt.title(nom_dataset)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(avec_grid)
    plt.show()


# plot_frontiere:
def plot_frontiere(desc_set, label_set, classifier, step=30):
    """desc_set * label_set * Classifier * int -> NoneType
    Remarque: le 4e argument est optionnel et donne la "résolution" du tracé: plus il est important
    et plus le tracé de la frontière sera précis.
    Cette fonction affiche la frontière de décision associée au classifieur
    """
    mmax = desc_set.max(0)
    mmin = desc_set.min(0)
    x1grid, x2grid = np.meshgrid(
        np.linspace(mmin[0], mmax[0], step), np.linspace(mmin[1], mmax[1], step)
    )
    grid = np.hstack((x1grid.reshape(x1grid.size, 1), x2grid.reshape(x2grid.size, 1)))

    # calcul de la prediction pour chaque point de la grille
    res = np.array([classifier.predict(grid[i, :]) for i in range(len(grid))])
    res = res.reshape(x1grid.shape)
    # tracer des frontieres
    # colors[0] est la couleur des -1 et colors[1] est la couleur des +1
    plt.contourf(
        x1grid, x2grid, res, colors=["darksalmon", "skyblue"], levels=[-1000, 0, 1000]
    )


def genere_train_test(desc_set, label_set, n_pos, n_neg):
    """permet de générer une base d'apprentissage et une base de test
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


def plot2DTrainTestSet(
    d_train, l_train, d_test, l_test, nom_dataset="Dataset", avec_grid=True
):
    """array * array array * array * str * bool-> affichage
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

    idx_neg_test = l_test == -1
    idx_pos_test = l_test == 1
    desc_neg_test = d_test[idx_neg_test]
    desc_pos_test = d_test[idx_pos_test]

    idx_neg_train = l_train == -1
    idx_pos_train = l_train == 1
    desc_neg_train = d_train[idx_neg_train]
    desc_pos_train = d_train[idx_pos_train]
    plt.scatter(
        desc_neg_test[:, 0],
        desc_neg_test[:, 1],
        marker="o",
        color="yellow",
        label="classe -1 test",
    )
    plt.scatter(
        desc_pos_test[:, 0],
        desc_pos_test[:, 1],
        marker="x",
        color="green",
        label="classe +1 test",
    )
    plt.scatter(
        desc_neg_train[:, 0],
        desc_neg_train[:, 1],
        marker="o",
        color="red",
        label="classe -1",
    )
    plt.scatter(
        desc_pos_train[:, 0],
        desc_pos_train[:, 1],
        marker="x",
        color="blue",
        label="classe +1",
    )
    plt.title(nom_dataset)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(avec_grid)
    plt.show()


def create_XOR(n, var):
    """int * float -> tuple[ndarray, ndarray]
    Hyp: n et var sont positifs
    n: nombre de points voulus
    var: variance sur chaque dimension
    """
    sigma = np.sqrt(var)
    n_par_nuag = n // 4

    gaus = np.random.normal(0, sigma, (n, 2))

    # nuage avec coord (1,1)
    gaus[0:n_par_nuag, 0] += 1
    gaus[0:n_par_nuag, 1] += 1

    # (-1, -1)
    gaus[n_par_nuag : 2 * n_par_nuag, 0] -= 1
    gaus[n_par_nuag : 2 * n_par_nuag, 1] -= 1

    # (1, -1)
    gaus[2 * n_par_nuag : 3 * n_par_nuag, 0] += 1
    gaus[2 * n_par_nuag : 3 * n_par_nuag, 1] -= 1

    # Центр (-1, 1)
    gaus[3 * n_par_nuag :, 0] -= 1
    gaus[3 * n_par_nuag :, 1] += 1

    labels = np.ones(n)
    labels[2 * n_par_nuag :] = -1

    return gaus, labels


class Distance(ABC):
    """Classe abstraite pour représenter des mesures de distances
    Elle permet de définir une hiérarchie pour les distances
    """

    def __init__(self, nom):
        """Constructeur:
        prend en argument le nom (str) de la distance créé
        """
        self.__nom: str = nom

    @abstractmethod
    def calcule(self, v, M):
        """Arguments:
            - v: un vecteur
            - M: un vecteur ou une matrice
        Hypothèse: v et M ont le même nombre de colonnes
        Retour:
            - un float si M est une vecteur: distance entre v et M
            - une np.series si M est une matrice: distances entre le vecteur v et chaque vecteur de M
        """
        # le calcul de distance dépend de la mesure que l'on utilise, il sera implémenté
        # dans les sous-classes de cette classe.
        pass

    def __str__(self) -> str:
        """rend une chaîne de caractères (méthode toString)
        Par exemple, pour afficher des informations sur l'objet
        """
        return "Distance " + self.__nom


def CHA_initialise(DF):
    """Initialise la partition de départ pour CHA
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
    """Fusionne les 2 groupes les plus proches selon le linkage
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

    # Recherche des 2 clusters les plus proches
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            id0 = ids[i]
            id1 = ids[j]
            G0 = df.loc[p0[id0]]
            G1 = df.loc[p0[id1]]
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
    """Algorithme de CHA
    Arguments:
        - df: DataFrame de la base d'apprentissage
        - linkage_instance: instance de la classe Linkage à utiliser pour calculer les distances entre groupes
        - verbose: affichage du debuggage
    Retour:
        - liste resultat: [[id0, id1, dist, taille], ...]
    """
    p = CHA_initialise(df)
    resultat = []

    while len(p) > 1:
        # La taille du cluster fusionne se calcule sur la partition avant fusion
        p1, id0, id1, dist = CHA_fusionne(df, p, linkage_instance, verbose)
        taille = len(p[id0]) + len(p[id1])
        resultat.append([id0, id1, float(dist), taille])
        p = p1

    return resultat


def CHA_dendrogramme(results, linkage_info):
    """Affiche le dendrogramme à partir des résultats de CHA
    Arguments:
        - results: liste des fusions effectuées par CHA (id0, id1, dist, taille)
        - linkage_info: str, informations sur le linkage utilisé (pour le titre du graphique)
    """

    # Paramètre de la fenêtre d'affichage:
    plt.figure(figsize=(30, 15))  # taille : largeur x hauteur
    plt.title(f"Dendrogramme - {linkage_info}", fontsize=25)
    plt.xlabel("Indice d'exemple", fontsize=25)
    plt.ylabel("Distance", fontsize=25)

    # Construction du dendrogramme pour notre clustering :
    scipy.cluster.hierarchy.dendrogram(
        results,
        leaf_font_size=24.0,  # taille des caractères de l'axe des X
    )

    # Affichage du résultat obtenu:
    plt.show()


class DistanceEuclidienne(Distance):
    """Classe de distance Euclidienne (Minkowski avec p=2)"""

    def __init__(self):
        """Constructeur"""
        super().__init__("Euclidienne")

    def calcule(self, v, M):
        """Arguments:
            - v: un vecteur
            - M: un vecteur ou une matrice
        Hypothèse: v et M ont le même nombre de colonnes
        Retour:
            - un float si M est une vecteur: distance entre v et M
            - une np.series si M est une matrice: distances entre le vecteur v et chaque vecteur de M
        """
        if v.ndim != 1:
            raise TypeError(
                "Argument incorrect: le premier argument doit être un vecteur"
            )

        # Cas vecteur
        if M.ndim == 1:
            return np.linalg.norm(M - v)

        # Cas matrice
        elif M.ndim == 2:
            return np.linalg.norm(M - v, axis=1)

        else:
            raise TypeError("Argument incorrect: M doit être un vecteur ou une matrice")


class DistanceMinkowski(Distance):
    """Classe de distance Minkowski"""

    def __init__(self, p=3):
        """Constructeur
        Argument:
            - p: ordre de la distance de Minkowski (p>=1)
        """
        if p < 1:
            raise ValueError("L'ordre p doit être supérieur ou égal à 1")
        super().__init__(f"Minkowski(p={p})")
        self.__p = p

    def calcule(self, v, M):
        """Arguments:
            - v: un vecteur
            - M: un vecteur ou une matrice
        Hypothèse: v et M ont le même nombre de colonnes
        Retour:
            - un float si M est une vecteur: distance entre v et M
            - une np.series si M est une matrice: distances entre le vecteur v et chaque vecteur de M
        """
        if v.ndim != 1:
            raise TypeError(
                "Argument incorrect: le premier argument doit être un vecteur"
            )

        # Cas vecteur
        if M.ndim == 1:
            return np.sum(np.abs(M - v) ** self.__p) ** (1 / self.__p)

        # Cas matrice
        elif M.ndim == 2:
            return np.sum(np.abs(M - v) ** self.__p, axis=1) ** (1 / self.__p)

        else:
            raise TypeError("Argument incorrect: M doit être un vecteur ou une matrice")
