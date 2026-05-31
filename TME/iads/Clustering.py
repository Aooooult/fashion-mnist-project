# -*- coding: utf-8 -*-

"""
Package: iads
File: Clustering.py
Année: LU3IN026 - semestre 2 - 2025-2026, Sorbonne Université
"""

# ---------------------------
# Fonctions de Clustering

# import externe
import numpy as np
import pandas as pd

# ------------------------

from abc import ABC, abstractmethod


class Linkage(ABC):
    """Classe abstraite pour représenter des approches Linkage"""

    def __init__(self, nom):
        """Constructeur:
        prend en argument:
            - nom (str) du linkage
        """
        self.__nom: str = nom

    @abstractmethod
    def calcule(self, G1, G2, verbose=False):
        """Arguments:
            - G1 et G2 sont des dataframes ou des np.array
            - verbose: pour afficher des messages de débuggage si besoin
        Hypothèse:
            - G1 et G2 ont le même nombre de colonnes
        Retour:
            - la distance entre G1 et G2 selon le linkage
        """
        pass

    def __str__(self) -> str:
        """rend une chaîne de caractères (méthode toString)
        Par exemple, pour afficher des informations sur l'objet
        """
        return "Linkage " + self.__nom


class LinkageComplete(Linkage):
    """Classe pour le linkage "Complete" """

    def __init__(self, distance=DistanceEuclidienne()):
        """Constructeur:
        prend en argument:
            - nom (str) du linkage
            - distance (Distance): mesure de distance entre 2 exemples
              par défaut: distance euclidienne
        """
        super().__init__("complete")
        self.__distance = distance

    def calcule(self, G1, G2, verbose=False):
        """Arguments:
            - G1 et G2 sont des dataframes ou des np.array
            - verbose: pour afficher des messages de débuggage si besoin
        Hypothèse:
            - G1 et G2 ont le même nombre de colonnes
        Retour:
            - la distance entre G1 et G2 selon le linkage
        """

        G1 = np.atleast_2d(G1)
        G2 = np.atleast_2d(G2)

        max_dist = -np.inf
        for row in G1:
            distances = self.__distance.calcule(row, G2)
            distances = np.atleast_1d(distances)
            max_dist = max(max_dist, np.max(distances))

        if verbose:
            print(f"Distance: {max_dist}")
        return max_dist

    def __str__(self) -> str:
        """rend une chaîne de caractères (méthode toString)
        Par exemple, pour afficher des informations sur l'objet
        """
        return super().__str__() + " (" + self.__distance.__str__() + ")"


class LinkageSimple(Linkage):
    """Classe pour le linkage "Simple" """

    def __init__(self, distance=DistanceEuclidienne()):
        """Constructeur:
        prend en argument:
            - nom (str) du linkage
            - distance (Distance): mesure de distance entre 2 exemples
              par défaut: distance euclidienne
        """
        super().__init__("simple")
        self.__distance = distance

    def calcule(self, G1, G2, verbose=False):
        """Arguments:
            - G1 et G2 sont des dataframes ou des np.array
            - verbose: pour afficher des messages de débuggage si besoin
        Hypothèse:
            - G1 et G2 ont le même nombre de colonnes
        Retour:
            - la distance entre G1 et G2 selon le linkage
        """

        G1 = np.atleast_2d(G1)
        G2 = np.atleast_2d(G2)

        min_dist = np.inf
        for row in G1:
            distances = self.__distance.calcule(row, G2)
            distances = np.atleast_1d(distances)
            min_dist = min(min_dist, np.min(distances))

        if verbose:
            print(f"Distance: {min_dist}")

        return min_dist


# ------------------------------------------------------
class LinkageAverage(Linkage):
    """Classe pour le linkage "Average" """

    def __init__(self, distance=DistanceEuclidienne()):
        """Constructeur:
        prend en argument:
            - nom (str) du linkage
            - distance (Distance): mesure de distance entre 2 exemples
              par défaut: distance euclidienne
        """
        super().__init__("average")
        self.__distance = distance

    def calcule(self, G1, G2, verbose=False):
        """Arguments:
            - G1 et G2 sont des dataframes ou des np.array
            - verbose: pour afficher des messages de débuggage si besoin
        Hypothèse:
            - G1 et G2 ont le même nombre de colonnes
        Retour:
            - la distance entre G1 et G2 selon le linkage
        """

        G1 = np.atleast_2d(G1)
        G2 = np.atleast_2d(G2)
        # print(f"DataFrame:{G1.size}")
        moyenne_dist = []
        for row in G1:
            moyenne_dist.append(np.average(self.__distance.calcule(row, G2)))

        moyenne_dist = sum(moyenne_dist) / len(moyenne_dist)
        if verbose:
            print(f"Distance: {moyenne_dist}")
        return moyenne_dist


# ------------------------------------------------------
class LinkageCentroide(Linkage):
    """Classe pour le linkage "Centroide" """

    def __init__(self, distance=DistanceEuclidienne()):
        """Constructeur:
        prend en argument:
            - nom (str) du linkage
            - distance (Distance): mesure de distance entre 2 exemples
              par défaut: distance euclidienne
        """
        super().__init__("centroide")
        self.__distance = distance

    ########################
    ############ A COMPLETER
    ########################
    def calcule(self, G1, G2, verbose=False):
        """Arguments:
            - G1 et G2 sont des dataframes ou des np.array
            - verbose: pour afficher des messages de débuggage si besoin
        Hypothèse:
            - G1 et G2 ont le même nombre de colonnes
        Retour:
            - la distance entre G1 et G2 selon le linkage
        """

        G1 = np.atleast_2d(G1)
        G2 = np.atleast_2d(G2)

        # Calcul des centroïdes (toujours !)
        avg_G1 = np.mean(G1, axis=0)
        avg_G2 = np.mean(G2, axis=0)

        # Distance entre centroïdes via la mesure fournie
        dist = self.__distance.calcule(avg_G1, avg_G2)

        if verbose:
            print(f"Distance: {dist}")

        return dist


# ------------------------------------------------------



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

# --- Code ajouté automatiquement ---

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




class KMoyennes():
    """ Classe implémentant l'algorithme des K-moyennes
    """
    def __init__(self, K, distance=DistanceEuclidienne() ):
        """ Argument:
                - K (int) : nombre de clusters voulus
                - distance (Distance): mesure de distance entre 2 exemples
                  par défaut: distance euclidienne
        """
        if K<1:
            raise TypeError("KMoyennes: K doit être strictement plus grand que 0 !")
        self.__K = K 
        self.__distance = distance

    def get_K(self):
        """ Accesseur de la variable __K
        """
        return self.__K

    def __str__(self) -> str:
        """ rend une chaîne de caractères (méthode toString)
            Par exemple, pour afficher des informations sur l'objet
        """
        return f"(K={self.__K}, {self.__distance})"
 
    def inertie_cluster(self,Ens):
        """ Arguments :
                - Ens: array qui représente un cluster
            Hypothèse: len(Ens)> >= 2
            L'inertie est la somme (au carré) des distances des points au centroide.
        """

        Ens = np.asarray(Ens)
        centre = np.mean(Ens, axis=0)
        return np.sum((Ens - centre) ** 2)
        
    def init(self,Ens):
        """ Argument :
                - Ens: Array contenant n exemples
        """
        Ens = np.asarray(Ens)
        Ens_copy = Ens.copy()
        np.random.shuffle(Ens_copy)
        self.__centroides = Ens_copy[:self.__K]
        return self.__centroides

    def plus_proche(self,exemple,Centres):
        """ Arguments :
                - exemple : Array contenant un exemple
                - Centres : Array contenant les K centres
        """
        
        exemple = np.asarray(exemple)
        Centres = np.asarray(Centres)
    
        distances = np.linalg.norm(Centres - exemple, axis=1)
        return np.argmin(distances)


    def affecte_cluster(self,Base,Centres):
        """ Arguments :
                - Base: Array contenant la base d'apprentissage
                - Centres : Array contenant des centroides
        """
        
        Base = np.asarray(Base)
        Centres = np.asarray(Centres)
    
        new_clusters = {i: [] for i in range(self.__K)}
    
        for id_point, point in enumerate(Base):
            distances = np.linalg.norm(Centres - point, axis=1)
            centre = np.argmin(distances)
            new_clusters[centre].append(id_point)
    
        return new_clusters
    
    def centroides(self,Base, U):
        """ Arguments :
                - Base : Array contenant la base d'apprentissage
                - U : Dictionnaire d'affectation
        """


        Base = np.asarray(Base)
        n, d = Base.shape
    
        centroids = np.zeros((self.__K, d))
    
        for j in range(self.__K):
            indices = U[j]
    
            if len(indices) == 0:
                centroids[j] = Base[np.random.randint(0, n)]
            else:
                centroids[j] = np.mean(Base[indices], axis=0)
    
        return centroids

    
    def inertie_globale(self,Base, U):
        """ Arguments :
                - Base : Array pour la base d'apprentissage
                - U : Dictionnaire d'affectation
        """
        
        Base = np.asarray(Base)
        inertie = 0.0
    
        for j in range(self.__K):
            indices = U[j]
            points = Base[indices]
    
            if len(points) > 0:
                inertie += self.inertie_cluster(points)
    
        return inertie
        
    def train(self,Base, epsilon, iter_max, verbose=False):
        """ Arguments :
                - Base : Array pour la base d'apprentissage
                - epsilon : réel >0
                - iter_max : entier >1
                - verbose: pour afficher des messages de débuggage si besoin
        """

        # 1. initialisation des centres
        centres = self.init(Base)
    
        # 2. première affectation
        U = self.affecte_cluster(Base, centres)
    
        inertie = self.inertie_globale(Base, U)
    
        for ite in range(1, iter_max):
    
            # 3. recalcul des centres
            centres = self.centroides(Base, U)
    
            # 4. nouvelle affectation
            new_U = self.affecte_cluster(Base, centres)
    
            # 5. nouvelle inertie
            new_inertie = self.inertie_globale(Base, new_U)

            if abs(inertie - new_inertie) < epsilon:
                return centres, new_U
    
            # 7. mise à jour
            U = new_U
            inertie = new_inertie
    
        return centres, U
       
