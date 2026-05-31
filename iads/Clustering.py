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
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod
from sklearn.metrics import confusion_matrix
from iads.utils import DistanceEuclidienne
# ------------------------ 


class Linkage(ABC):
    """ Classe abstraite pour représenter des approches Linkage
    """
    def __init__(self,nom):
        """ Constructeur:
            prend en argument:
                - nom (str) du linkage
        """
        self.__nom: str = nom

    @abstractmethod
    def calcule(self, G1, G2, verbose= False):
        """ Arguments:
                - G1 et G2 sont des dataframes ou des np.array
                - verbose: pour afficher des messages de débuggage si besoin
            Hypothèse:
                - G1 et G2 ont le même nombre de colonnes
            Retour:
                - la distance entre G1 et G2 selon le linkage
        """
        pass

    def __str__(self) -> str:
        """ rend une chaîne de caractères (méthode toString)
            Par exemple, pour afficher des informations sur l'objet
        """
        return "Linkage "+self.__nom

class LinkageComplete(Linkage):
    """ Classe pour le linkage "Complete"
    """
    def __init__(self,distance=DistanceEuclidienne()):
        """ Constructeur:
            prend en argument:
                - nom (str) du linkage
                - distance (Distance): mesure de distance entre 2 exemples
                  par défaut: distance euclidienne
        """
        super().__init__("complete")
        self.__distance = distance

    def calcule(self, G1, G2,verbose=False):
        """ Arguments:
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

        if (verbose):
            print(f"Distance: {max_dist}")
        return max_dist


    def __str__(self) -> str:
        """ rend une chaîne de caractères (méthode toString)
            Par exemple, pour afficher des informations sur l'objet
        """
        return super().__str__()+ " ("+self.__distance.__str__()+")"

class LinkageSimple(Linkage):
    """ Classe pour le linkage "Simple"
    """
    def __init__(self,distance=DistanceEuclidienne()):
        """ Constructeur:
            prend en argument:
                - nom (str) du linkage
                - distance (Distance): mesure de distance entre 2 exemples
                  par défaut: distance euclidienne
        """
        super().__init__("simple")
        self.__distance = distance

    def calcule(self, G1, G2,verbose=False):
        """ Arguments:
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
    """ Classe pour le linkage "Average"
    """
    def __init__(self,distance=DistanceEuclidienne()):
        """ Constructeur:
            prend en argument:
                - nom (str) du linkage
                - distance (Distance): mesure de distance entre 2 exemples
                  par défaut: distance euclidienne
        """
        super().__init__("average")
        self.__distance = distance

    def calcule(self, G1, G2,verbose=False):
        """ Arguments:
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

        moyenne_dist  = sum(moyenne_dist) / len(moyenne_dist)
        if (verbose):
            print(f"Distance: {moyenne_dist}")
        return moyenne_dist


# ------------------------------------------------------
class LinkageCentroide(Linkage):
    """ Classe pour le linkage "Centroide"
    """
    def __init__(self,distance=DistanceEuclidienne()):
        """ Constructeur:
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
    def calcule(self, G1, G2,verbose=False):
        """ Arguments:
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
        
        if verbose:
            print(f"K-Moyennes: Initialisation (inertie = {inertie:1.4f})")
    
        for ite in range(1, iter_max):
    
            # 3. recalcul des centres
            centres = self.centroides(Base, U)
    
            # 4. nouvelle affectation
            new_U = self.affecte_cluster(Base, centres)
    
            # 5. nouvelle inertie
            new_inertie = self.inertie_globale(Base, new_U)
            
            if verbose:
                print(f"K-Moyennes (ite {ite}/{iter_max}): inertie = {new_inertie:1.4f}, diff = {abs(inertie - new_inertie):1.4f}")

            if abs(inertie - new_inertie) < epsilon:
                if verbose:
                    print(f"Convergence atteinte à l'itération {ite} !")
                return centres, new_U
    
            # 7. mise à jour
            U = new_U
            inertie = new_inertie
    
        if verbose:
            print(f"Fin des {iter_max} itérations (sans convergence totale).")
        return centres, U
       
