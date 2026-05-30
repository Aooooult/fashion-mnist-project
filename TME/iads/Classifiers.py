# -*- coding: utf-8 -*-

"""
Package: iads
File: Classifiers.py
Année: LU3IN026 - semestre 2 - 2025-2026, Sorbonne Université
"""

# Classfieurs implémentés en LU3IN026
# Version de départ : Février 2026

# Import de packages externes
import numpy as np
import pandas as pd

import copy

from abc import ABC, abstractmethod


# ---------------------------
class Classifier(ABC):
    """Classe (abstraite) pour représenter un classifieur
    Attention: cette classe est ne doit pas être instanciée.
    """

    __nombre_crees: int = (
        0  # Variable de classe pour compter le nombre de classifiers créés
    )

    def __init__(self, input_dimension):
        """Constructeur de Classifier
        Argument:
            - intput_dimension (int) : dimension de la description des exemples
        Hypothèse : input_dimension > 0
        """
        Classifier.__nombre_crees += 1
        self.__ident = Classifier.__nombre_crees  # identifiant du classifieur (unique)
        self.__dimension = input_dimension

    def get_dimension(self):
        """Accesseur de la variable __dimension"""
        return self.__dimension

    def __str__(self) -> str:
        """rend une chaîne de caractères (méthode toString)
        Par exemple, pour afficher des informations sur l'objet
        """
        return f"Classifier #{self.__ident} (d{self.__dimension})"

    @abstractmethod
    def train(self, desc_set, label_set) -> None:
        """Permet d'entrainer le modele sur l'ensemble donné
        desc_set: array avec des descriptions
        label_set: array avec les labels correspondants
        Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        pass

    @abstractmethod
    def score(self, x) -> float:
        """rend le score de prédiction sur x (valeur réelle)
        x: une description
        """
        pass

    @abstractmethod
    def predict(self, x) -> int:
        """rend la prediction sur x (soit -1 ou soit +1)
        x: une description
        """
        pass

    def accuracy(self, desc_set, label_set) -> float:
        """rend le taux d'exemples bien classés dans le dataset
        desc_set: array avec des descriptions
        label_set: array avec les labels correspondants
        Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        corr = 0
        nb_donnees = len(desc_set)
        for i in range(nb_donnees):
            if label_set[i] == self.predict(desc_set[i]):
                corr += 1

        return corr / nb_donnees


class ClassifierKNN(Classifier):
    """Classe pour représenter un classifieur par K plus proches voisins.
    Cette classe hérite de la classe Classifier
    """

    def __init__(self, input_dimension, k):
        """Constructeur de Classifier
        Argument:
            - intput_dimension (int) : dimension d'entrée des exemples
            - k (int) : nombre de voisins à considérer
        Hypothèse : input_dimension > 0
        """
        super().__init__(input_dimension)  # Appel du constructeur de la classe mère
        self.__k = k
        # les 2 variables suivantes seront utilisées dans la méthode train()
        self.__desc_set = None
        self.__labels_set = None

    def __str__(self) -> str:
        """rend une chaîne de caractères (méthode toString)
        Par exemple, pour afficher des informations sur l'objet
        """
        return f"ClassifierKNN (dim: {self.get_dimension()}, k: {self.__k})"
        # raise NotImplementedError("Please Implement this method")

    def train(self, desc_set, label_set) -> None:
        """Permet d'entrainer le modele sur l'ensemble donné
        desc_set: array avec des descriptions
        label_set: array avec les labels correspondants
        Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        self.__desc_set = desc_set
        self.__labels_set = label_set
        # raise NotImplementedError("Please Implement this method")

    def score(self, x) -> float:
        """rend la proportion de +1 parmi les k ppv de x (valeur réelle)
        x: une description : un array
        """
        distances = np.linalg.norm(self.__desc_set - x, axis=1)
        indices_k_proches = np.argsort(distances)[: self.__k]
        voisins_labels = self.__labels_set[indices_k_proches]
        proportion_pos = np.sum(voisins_labels == 1) / self.__k
        return 2 * (proportion_pos - 0.5)
        # raise NotImplementedError("Please Implement this method")

    def predict(self, x) -> int:
        """rend la prediction sur x (-1 ou +1)
        x: une description : un array
        """
        return 1 if self.score(x) > 0 else -1
        # raise NotImplementedError("Please Implement this method")


class ClassifierLineaireRandom(Classifier):
    """Classe pour représenter un classifieur linéaire aléatoire
    Cette classe hérite de la classe Classifier
    """

    def __init__(self, input_dimension):
        """Constructeur de Classifier
        Argument:
            - input_dimension (int) : dimension de la description des exemples
        Hypothèse : input_dimension > 0
        """
        super().__init__(input_dimension)
        v = np.random.uniform(-1, 1, input_dimension)
        norma = np.linalg.norm(v)
        self.__w = v / norma

    def __str__(self) -> str:
        """rend une chaîne de caractères (méthode toString)
        Par exemple, pour afficher des informations sur l'objet
        """
        return f"Classifier (dim: {self.get_dimension()}, w: {self.__w})"
        # raise NotImplementedError("Please Implement this method")

    def train(self, desc_set, label_set) -> None:
        """Permet d'entrainer le modele sur l'ensemble donné
        desc_set: array avec des descriptions
        label_set: array avec les labels correspondants
        Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        print("Pas d'apprentissage pour ce classifieur")
        # raise NotImplementedError("Please Implement this method")

    def score(self, x):
        return np.dot(x, self.__w)

    def predict(self, x):
        return 1 if self.score(x) >= 0 else -1


# ------------------------ A COMPLETER : DEFINITION DU CLASSIFIEUR PERCEPTRON


class ClassifierPerceptronTME3(Classifier):
    """Perceptron de Rosenblatt"""

    def __init__(self, input_dimension, learning_rate=0.01, init=True, verbose=False):
        """Constructeur de Classifier
        Argument:
            - input_dimension (int) : dimension de la description des exemples (>0)
            - learning_rate (par défaut 0.01): epsilon
            - init est le mode d'initialisation de w:
                - si True (par défaut): initialisation à 0 de w,
                - si False : initialisation par tirage aléatoire de valeurs petites
            - verbose: pour dire si on veut afficher la valeur d'initialisation
        """
        super().__init__(input_dimension)  # Appel du constructeur de la classe mère
        ########### A COMPLETER ###################
        self.__learning = learning_rate
        if init:
            self.__w = np.zeros(input_dimension)
        else:
            self.__w = (2 * np.random.rand(input_dimension) - 1) * 0.001

        self.__allw = [self.__w.copy()]
        ###########################################
        if verbose:
            print(
                f"TME3{super().__str__()}: initialisation (learning rate= {self.__learning}) w= {self.__w}"
            )

    def train_step(self, desc_set, label_set, stabilised=False):
        """Réalise une unique itération sur tous les exemples du dataset
        donné en prenant les exemples aléatoirement.
        Arguments:
            - desc_set: array avec des descriptions
            - label_set: array avec les labels correspondants
        """
        rng = np.random.default_rng()
        idx = list(range(len(label_set)))
        rng.shuffle(idx)

        for i in idx:
            x_i = desc_set[i]
            y_i = label_set[i]
            y = self.score(x_i)
            # print(f"score: {y}")
            if stabilised:
                # print("st")
                if y * y_i < 1:
                    self.__w += self.__learning * (y_i - y) * x_i
                    self.__allw.append(self.__w.copy())
                # print(self.__allw)
            else:
                if (y_i * y) <= 0:
                    self.__w = self.__w + self.__learning * y_i * x_i
                    self.__allw.append(self.__w.copy())

    def __str(self) -> str:
        """rend une chaîne de caractères (méthode toString)
        Par exemple, pour afficher des informations sur l'objet
        """
        return f"ClassifierPerceptronTME3 (d{self.get_dimension()})"

    def train(
        self,
        desc_set,
        label_set,
        nb_max=100,
        seuil=0.001,
        stabilised=False,
        verbose=False,
    ):
        """Apprentissage itératif du perceptron sur le dataset donné.
        Arguments:
            - desc_set: array avec des descriptions
            - label_set: array avec les labels correspondants
            - nb_max (par défaut: 100) : nombre d'itérations maximale
            - seuil (par défaut: 0.001) : seuil de convergence
            - verbose (par défaut: False): affichage de messages d'information
        Retour: la fonction rend une liste
            - liste des valeurs de norme de différences
        """
        result_list = list()
        for i in range(nb_max):
            w_old = self.__w.copy()
            self.train_step(desc_set, label_set, stabilised)
            w_new = self.__w
            dif = np.linalg.norm(np.abs(w_new - w_old))
            result_list.append(dif)
            if dif < seuil:
                return result_list
        return result_list

    def score(self, x):
        """rend le score de prédiction sur x (valeur réelle)
        x: une description
        """
        return np.dot(self.__w, x)

    def predict(self, x):
        """rend la prediction sur x (soit -1 ou soit +1)
        x: une description
        """
        return 1 if (self.score(x)) >= 0 else -1

    def get_allw(self):
        return self.__allw


class ClassifierPerceptron(Classifier):
    """Perceptron de Rosenblatt"""

    def __init__(self, input_dimension, learning_rate=0.01, init=True, verbose=False):
        """Constructeur de Classifier
        Argument:
            - input_dimension (int) : dimension de la description des exemples (>0)
            - learning_rate (par défaut 0.01): epsilon
            - init est le mode d'initialisation de w:
                - si True (par défaut): initialisation à 0 de w,
                - si False : initialisation par tirage aléatoire de valeurs petites
            - verbose: pour dire si on veut afficher la valeur d'initialisation
        """
        super().__init__(input_dimension)  # Appel du constructeur de la classe mère
        ########### A COMPLETER ###################
        self.__learning = learning_rate
        print(f"input_dimension: {input_dimension}")
        if init:
            self.__w = np.zeros(input_dimension + 1)
        else:
            self.__w = (2 * np.random.rand(input_dimension) - 1) * 0.001
            # print(f"w: {self.__w}")
            self.__w = np.append(self.__w, np.random.rand())
            # print(f"w: {self.__w}")
            # print(f"input_dimension_after: {self.__w.shape}")
        ###########################################
        self.__allw = [self.__w.copy()]
        if verbose:
            print(
                f"{super().__str__()}: initialisation (learning rate= {self.__learning}) w= {self.__w}"
            )

    def train_step(self, desc_set, label_set, stabilised=False):
        """Réalise une unique itération sur tous les exemples du dataset
        donné en prenant les exemples aléatoirement.
        Arguments:
            - desc_set: array avec des descriptions
            - label_set: array avec les labels correspondants
            - stabilised: boolean pour dire si on veut la variante stabilisée (par défaut: False)
        """
        rng = np.random.default_rng()
        idx = list(range(len(label_set)))
        rng.shuffle(idx)
        desc_set_copy = copy.deepcopy(desc_set)
        desc_set_copy = ClassifierPerceptron.augmente(desc_set_copy)

        for i in idx:
            x_i = desc_set[i]
            y_i = label_set[i]
            x_i_aug = desc_set_copy[i]
            y = self.score(x_i)
            if stabilised:
                if y * y_i < 1:
                    self.__w += self.__learning * (y_i - y) * x_i_aug
                    self.__allw.append(self.__w.copy())

            else:
                if (y_i * y) <= 0:
                    self.__w = self.__w + self.__learning * y_i * x_i_aug
                    self.__allw.append(self.__w.copy())

    def __str__(self) -> str:
        """rend une chaîne de caractères (méthode toString)
        Par exemple, pour afficher des informations sur l'objet
        """
        return f"ClassifierPerceptron (d{self.get_dimension()})"

    def train(
        self,
        desc_set,
        label_set,
        nb_max=100,
        seuil=0.001,
        stabilised=False,
        verbose=False,
    ):
        """Apprentissage itératif du perceptron sur le dataset donné.
        Arguments:
            - desc_set: array avec des descriptions
            - label_set: array avec les labels correspondants
            - nb_max (par défaut: 100) : nombre d'itérations maximale
            - seuil (par défaut: 0.01) : seuil de convergence
            - stabilised: boolean pour dire si on veut la variante stabilisée (par défaut: False)
            - verbose (par défaut: False): affichage de messages d'information
        Retour: la fonction rend une liste
            - liste des valeurs de norme de différences
        """
        result_list = list()
        for i in range(nb_max):
            w_old = self.__w.copy()
            self.train_step(desc_set, label_set, stabilised)
            w_new = self.__w
            dif = np.linalg.norm(np.abs(w_new - w_old))
            result_list.append(dif)
            if dif < seuil:
                return result_list
        return result_list

    def score(self, x):
        """rend le score de prédiction sur x (valeur réelle)
        x: une description
        """
        x_app = ClassifierPerceptron.augmente(x)
        # print(f"x: {x_app}")
        # print(f"w: {self.__w}")
        return np.dot(self.__w, x_app)

    def predict(self, x):
        """rend la prediction sur x (soit -1 ou soit +1)
        x: une description
        """
        return 1 if (self.score(x)) >= 0 else -1

    def augmente(x):
        return np.insert(x, x.shape[-1], -1, axis=-1)

    def get_allw(self):
        return self.__allw


def shannon(P, k=2):
    """list[Number] * int -> float
    Hypothèse: P est une distribution de probabilités et k>0
    - P: distribution de probabilités
    - k: base du logarithme à utiliser, par défaut 2
    rend la valeur de l'entropie de Shannon correspondante
    """
    ########################## COMPLETER ICI
    logs = np.emath.logn(k, P)
    logs[logs == -np.inf] = 0
    HS = -np.sum(P * logs)
    return HS
    ##########################


def entropie(Y, k=2):
    """Y : (array) : ensemble de labels de classe
    Hypothèse: k>0
    - k: base du logarithme à utiliser, par défaut 2
    rend l'entropie de l'ensemble Y
    """
    ########################## COMPLETER ICI
    nb_elem = len(Y)
    if nb_elem == 0:
        return 0.0

    classes = np.unique(Y)
    freq = np.zeros(len(classes))
    for i, c in enumerate(classes):
        nb = len(Y[Y == c])
        freq[i] = nb / nb_elem

    logs = np.zeros_like(freq, dtype=float)
    mask = freq > 0
    logs[mask] = np.emath.logn(k, freq[mask])
    return -np.sum(freq[mask] * logs[mask])

    ##########################


def classe_majoritaire(Y):
    """Y : (array) : array de labels
    rend la classe majoritaire ()
    """
    classes = np.unique(Y)
    maximum = 0
    label = None
    for i in classes:
        nb = len(Y[Y == i])
        if nb > maximum:
            label = i
            maximum = nb
    return label


def construit_AD(X,Y,epsilon,LNoms = [], verbose=False):
    """ X,Y : dataset
        epsilon : seuil d'entropie pour le critère d'arrêt
        LNoms : liste des noms de features (colonnes) de description
    """

    entropie_ens = entropie(Y)
    if verbose:
        print(f"Construction: entropie classe {entropie_ens:1.5f}")
    if (entropie_ens <= epsilon):
        # ARRET : on crée une feuille
        noeud = NoeudCategoriel(-1,"Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
        if verbose:
            print("\tajout d'une feuille avec la classe {classe_majoritaire(Y)}")
    else:
        min_entropie = 1.1
        i_best = -1
        Xbest_valeurs = None

        #############

        # COMPLETER CETTE PARTIE : ELLE DOIT PERMETTRE D'OBTENIR DANS
        # i_best : le numéro de l'attribut qui minimise l'entropie
        # min_entropie : la valeur de l'entropie minimale
        # Xbest_valeurs : la liste des valeurs que peut prendre l'attribut i_best
        #
        # Il est donc nécessaire ici de parcourir tous les attributs et de calculer
        # la valeur de l'entropie de la classe pour chaque attribut.

        for j in range(X.shape[1]):  # pour chaque attribut X_j
            X_j = X[:, j]
            valeurs = np.unique(X_j)

            Hs = 0
            n = len(Y)

            for v in valeurs:
                index = np.where(X_j == v)
                Y_subset = Y[index]
                p_v = len(Y_subset) / n
                Hs += p_v * entropie(Y_subset)

                if Hs < min_entropie:
                    min_entropie = Hs
                    i_best = j
                    Xbest_valeurs = valeurs

        #############################################
        if verbose:
            nom = str(i_best)
            if LNoms != []:
                nom = LNoms[i_best]
            print(f"\tMeilleur: {nom}: entropie= {min_entropie:1.5f}")
        if len(LNoms)>0:  # si on a des noms de features
            noeud = NoeudCategoriel(i_best,LNoms[i_best])
        else:
            noeud = NoeudCategoriel(i_best)
        for v in Xbest_valeurs:
            if verbose:
                print(f"\mble des exemples de qui possède la valeur ainsi que l'ensemble de leurs labels. 3.2. calculer l'entropie conditionnelle de Shannon de la classe relativement à l'attribut . On note cette entropie tdescente pour {v}")
            noeud.ajoute_fils(v,construit_AD(X[X[:,i_best]==v], Y[X[:,i_best]==v],epsilon,LNoms,verbose))
    return noeud

class NoeudCategoriel:
    """ Classe pour représenter des noeuds d'un arbre de décision
    """
    def __init__(self, num_att=-1, nom=''):
        """ Constructeur: il prend en argument
            - num_att (int) : le numéro de l'attribut auquel il se rapporte: de 0 à ...
              si le noeud se rapporte à la classe, le numéro est -1, on n'a pas besoin
              de le préciser
            - nom (str) : une chaîne de caractères donnant le nom de l'attribut si
              il est connu (sinon, on ne met rien et le nom sera donné de façon
              générique: "att_Numéro")
        """
        self.__attribut = num_att    # numéro de l'attribut
        if (nom == ''):            # son nom si connu
            self.__nom_attribut = 'att_'+str(num_att)
        else:
            self.__nom_attribut = nom
        self.__Les_fils = None       # aucun fils à la création, ils seront ajoutés
        self.__classe   = None       # valeur de la classe si c'est une feuille

    def est_feuille(self):
        """ rend True si l'arbre est une feuille
            c'est une feuille s'il n'a aucun fils
        """
        return self.__Les_fils == None

    def ajoute_fils(self, valeur, Fils):
        """ valeur : valeur de l'attribut de ce noeud qui doit être associée à Fils
                     le type de cette valeur dépend de la base
            Fils (NoeudCategoriel) : un nouveau fils pour ce noeud
            Les fils sont stockés sous la forme d'un dictionnaire:
            Dictionnaire {valeur_attribut : NoeudCategoriel}
        """
        if self.__Les_fils == None:
            self.__Les_fils = dict()
        self.__Les_fils[valeur] = Fils
        # Rem: attention, on ne fait aucun contrôle, la nouvelle association peut
        # écraser une association existante.

    def ajoute_feuille(self,classe):
        """ classe: valeur de la classe
            Ce noeud devient un noeud feuille
        """
        self.__classe    = classe
        self.__Les_fils  = None   # normalement, pas obligatoire ici, c'est pour être sûr

    def classifie(self, exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple
            on rend la valeur None si l'exemple ne peut pas être classé (cf. les questions
            posées en fin de ce notebook)
        """
        if self.est_feuille():
            return self.__classe
        if exemple[self.__attribut] in self.__Les_fils:
            # descente récursive dans le noeud associé à la valeur de l'attribut
            # pour cet exemple:
            return self.__Les_fils[exemple[self.__attribut]].classifie(exemple)
        else:
            # Cas particulier : on ne trouve pas la valeur de l'exemple dans la liste des
            # fils du noeud... Voir la fin de ce notebook pour essayer de résoudre ce mystère...
            print('\t*** Warning: attribut ',self.__nom_attribut,' -> Valeur inconnue: ',exemple[self.__attribut])
            return None

    def compte_feuilles(self):
        """ rend le nombre de feuilles sous ce noeud
        """
        if self.est_feuille():
            return 1
        total = 0
        for noeud in self.__Les_fils:
            total += self.__Les_fils[noeud].compte_feuilles()
        return total

    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir l'afficher graphiquement
            Cette fonction ne nous intéressera pas plus que ça, elle ne sera donc pas expliquée
        """
        if self.est_feuille():
            g.node(prefixe,str(self.__classe),shape='box')
        else:
            g.node(prefixe, self.__nom_attribut)
            i =0
            for (valeur, sous_arbre) in self.__Les_fils.items():
                sous_arbre.to_graph(g,prefixe+str(i))
                g.edge(prefixe,prefixe+str(i), str(valeur))
                i = i+1
        return g

class ClassifierPerceptronStable(ClassifierPerceptron):
    """ Perceptron de Rosenblatt stabilisé
    """
    def __init__(self, input_dimension, learning_rate=0.01, init=True,verbose=False):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate (par défaut 0.01): epsilon
                - init est le mode d'initialisation de w:
                    - si True (par défaut): initialisation à 0 de w,
                    - si False : initialisation par tirage aléatoire de valeurs petites
                - verbose: pour dire si on veut afficher la valeur d'initialisation
        """
        super().__init__(input_dimension, learning_rate, init,verbose)  # Appel du constructeur de la classe mère
        if verbose:
            print(f"{super().__str__()} (stabilise)")

    def __str__(self) -> str:
        """ rend une chaîne de caractères (méthode toString)
            Par exemple, pour afficher des informations sur l'objet
        """
        return f"{super().__str__()} (stabilise)"
        ################### A COMPLETER
        #raise NotImplementedError("Vous devez implémenter cette fonction !")

    
    def train(self, desc_set, label_set, nb_max=100, seuil=0.001,stabilised=True, verbose=False):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: array avec des descriptions
                - label_set: array avec les labels correspondants
                - nb_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.01) : seuil de convergence
                - verbose (par défaut: False): affichage de messages d'information
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """
        return super().train(desc_set, label_set, nb_max, seuil,stabilised,verbose)
        ################### A COMPLETER        
        #raise NotImplementedError("Vous devez implémenter cette fonction !")
