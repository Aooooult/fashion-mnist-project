import numpy as np
import copy
import graphviz as gv
from iads import Classifiers as cl
import random


def classe_majoritaire(Y):
    classes = np.unique(Y)
    maximum = 0
    label = None
    for i in classes: 
        nb = len(Y[Y == i])
        if nb > maximum: 
            label = i
            maximum = nb
    return label

def shannon(P, k=2):
    logs = np.emath.logn(k, P)
    logs[logs == -np.inf] = 0
    HS = - np.sum(P * logs)
    return HS

def entropie(Y, k=2):
    if len(Y) == 0:
        return 0.0
    nb_elem = len(Y)
    classes = np.unique(Y)
    freq = np.zeros(len(classes))
    for i, c in enumerate(classes): 
        nb = len(Y[Y == c])
        freq[i] = nb / nb_elem
    return shannon(freq, k)

def discretise(m_desc, m_class, num_col, verbose=False):
    l_valeurs = np.unique(m_desc[:, num_col])
    if (len(l_valeurs) < 2):
        return ((None, float('Inf')), ([], []))
    best_seuil = None
    best_entropie = float('Inf')
    liste_entropies = []
    liste_coupures = []
    nb_exemples = len(m_class)
    for v in l_valeurs:
        cl_inf = m_class[m_desc[:, num_col] <= v]
        cl_sup = m_class[m_desc[:, num_col] > v]
        nb_inf = len(cl_inf)
        nb_sup = len(cl_sup)
        val_entropie_inf = entropie(cl_inf)
        val_entropie_sup = entropie(cl_sup)
        val_entropie = (nb_inf / float(nb_exemples)) * val_entropie_inf \
                       + (nb_sup / float(nb_exemples)) * val_entropie_sup
        liste_coupures.append(v)
        liste_entropies.append(val_entropie)
        if (best_entropie > val_entropie):
            best_entropie = val_entropie
            best_seuil = v
    return (best_seuil, best_entropie), (liste_coupures, liste_entropies)

def partitionne(m_desc, m_class, n, s):
    return ((m_desc[m_desc[:, n] <= s], m_class[m_desc[:, n] <= s]), \
            (m_desc[m_desc[:, n] > s], m_class[m_desc[:, n] > s]))

class NoeudCategoriel:
    def __init__(self, num_att=-1, nom=''):
        self.__attribut = num_att    
        if (nom == ''):            
            self.__nom_attribut = 'att_' + str(num_att)
        else:
            self.__nom_attribut = nom 
        self.__Les_fils = None       
        self.__classe   = None       
        
    def est_feuille(self):
        return self.__Les_fils == None
    
    def ajoute_fils(self, valeur, Fils):
        if self.__Les_fils == None:
            self.__Les_fils = dict()
        self.__Les_fils[valeur] = Fils
    
    def ajoute_feuille(self, classe):
        self.__classe    = classe
        self.__Les_fils  = None   
        
    def classifie(self, exemple):
        if self.est_feuille():
            return self.__classe
        if exemple[self.__attribut] in self.__Les_fils:
            return self.__Les_fils[exemple[self.__attribut]].classifie(exemple)
        else:
            return None
    
    def compte_feuilles(self):
        if self.est_feuille():
            return 1
        total = 0
        for noeud in self.__Les_fils:
            total += self.__Les_fils[noeud].compte_feuilles()
        return total
     
    def to_graph(self, g, prefixe='A'):
        if self.est_feuille():
            g.node(prefixe, str(self.__classe), shape='box')
        else:
            g.node(prefixe, self.__nom_attribut)
            i = 0
            for (valeur, sous_arbre) in self.__Les_fils.items():
                sous_arbre.to_graph(g, prefixe + str(i))
                g.edge(prefixe, prefixe + str(i), str(valeur))
                i = i + 1        
        return g

class NoeudNumerique:
    def __init__(self, num_att=-1, nom=''):
        self.__attribut = num_att    
        if (nom == ''):            
            self.__nom_attribut = 'att_' + str(num_att)
        else:
            self.__nom_attribut = nom
        self.__seuil = None          
        self.__Les_fils = None       
        self.__classe   = None       

    def est_feuille(self):
        return self.__Les_fils == None

    def ajoute_fils(self, val_seuil, fils_inf, fils_sup):
        if self.__Les_fils == None:
            self.__Les_fils = dict()
        self.__seuil = val_seuil
        self.__Les_fils['inf'] = fils_inf
        self.__Les_fils['sup'] = fils_sup

    def ajoute_feuille(self, classe):
        self.__classe    = classe
        self.__Les_fils  = None   

    def classifie(self, exemple):
        if self.est_feuille():
            return self.__classe if self.__classe is not None else 0
        if exemple is None or self.__seuil is None:
            return 0
        if self.__attribut < 0 or self.__attribut >= len(exemple):
            return 0
        if exemple[self.__attribut] <= self.__seuil:
            return self.__Les_fils['inf'].classifie(exemple)
        else:
            return self.__Les_fils['sup'].classifie(exemple)

    def compte_feuilles(self):
        if self.est_feuille():
            return 1
        return self.__Les_fils['inf'].compte_feuilles() + self.__Les_fils['sup'].compte_feuilles()

    def to_graph(self, g, prefixe='A'):
        if self.est_feuille():
            g.node(prefixe, str(self.__classe), shape='box')
        else:
            g.node(prefixe, str(self.__nom_attribut))
            self.__Les_fils['inf'].to_graph(g, prefixe + "g")
            self.__Les_fils['sup'].to_graph(g, prefixe + "d")
            g.edge(prefixe, prefixe + "g", '<=' + str(self.__seuil))
            g.edge(prefixe, prefixe + "d", '>' + str(self.__seuil))
        return g

def construit_AD(X, Y, epsilon, LNoms=[], verbose=False):
    entropie_ens = entropie(Y)
    if (entropie_ens <= epsilon):
        noeud = NoeudCategoriel(-1, "Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
    else:
        min_entropie = 1.1
        i_best = -1
        Xbest_valeurs = None
        
        for j in range(X.shape[1]):  
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
                    
        if (i_best != -1):
            if len(LNoms) > 0:  
                noeud = NoeudCategoriel(i_best, LNoms[i_best])    
            else:
                noeud = NoeudCategoriel(i_best)
            for v in Xbest_valeurs:
                noeud.ajoute_fils(v, construit_AD(X[X[:, i_best] == v], Y[X[:, i_best] == v], epsilon, LNoms, verbose))
        else:
            noeud = NoeudCategoriel(-1, "Label")
            noeud.ajoute_feuille(classe_majoritaire(Y))
    return noeud

def construit_AD_num(X, Y, epsilon, LNoms=[], verbose=False):
    (nb_lig, nb_col) = X.shape
    entropie_classe = entropie(Y)
    if (entropie_classe <= epsilon) or (nb_lig <= 1):
        noeud = NoeudNumerique(-1, "Label")
        classe_choisie = classe_majoritaire(Y)
        noeud.ajoute_feuille(classe_choisie)
    else:
        gain_max = 0.0  
        i_best = -1     
        Xbest_tuple = ((X, Y), (None, None))
        Xbest_seuil = None
        for i in range(nb_col):
            (seuil_i, entropie_i), _ = discretise(X, Y, i)
            if seuil_i is None:
                tuple_i = ((X, Y), (None, None))
                gain_info = 0.0
            else:
                tuple_i = partitionne(X, Y, i, seuil_i)
                gain_info = entropie_classe - entropie_i
            if gain_info > gain_max:
                gain_max = gain_info
                i_best = i
                Xbest_tuple = tuple_i
                Xbest_seuil = seuil_i
        if (i_best != -1): 
            if len(LNoms) > 0:  
                noeud = NoeudNumerique(i_best, LNoms[i_best])
            else:
                noeud = NoeudNumerique(i_best)
            ((left_data, left_class), (right_data, right_class)) = Xbest_tuple
            noeud.ajoute_fils(Xbest_seuil, \
                              construit_AD_num(left_data, left_class, epsilon, LNoms, verbose), \
                              construit_AD_num(right_data, right_class, epsilon, LNoms, verbose))
        else: 
            noeud = NoeudNumerique(-1, "Label")
            noeud.ajoute_feuille(classe_majoritaire(Y))
    return noeud

def construit_AD_gen(X, Y, epsilon, LNoms, LTypes, verbose=False):
    (nb_lig, nb_col) = X.shape
    entropie_classe = entropie(Y)
    if (entropie_classe <= epsilon) or (nb_lig <= 1):
        noeud = NoeudNumerique(-1, "Label")
        if len(Y) < 1:
            noeud.ajoute_feuille(None)
        else:
            noeud.ajoute_feuille(classe_majoritaire(Y))
    else:
        LCols_cat = [i for i in range(0, len(LTypes)) if not(LTypes[i])]
        if len(LCols_cat) == 0:
            return construit_AD_num(X, Y, epsilon, LNoms, verbose)
        gain_max = 0.0  
        i_best = -1     
        best_type_num = False
        best_seuil = None
        best_valeurs = None
        for i in range(0, X.shape[1]):
            if LTypes[i]:  
                (seuil_i, entropie_Xi), _ = discretise(X, Y, i)
                if seuil_i is None:
                    entropie_Xi = float('Inf')
            else:  
                valeurs = np.unique(X[:, i])
                entropie_Xi = 0.0
                for v in valeurs:
                    sous_Y = Y[X[:, i] == v]
                    entropie_Xi += (len(sous_Y) / float(len(Y))) * entropie(sous_Y)
            gain_info = entropie_classe - entropie_Xi
            if gain_info > gain_max:
                gain_max = gain_info
                i_best = i
                best_type_num = LTypes[i]
                if LTypes[i]:
                    best_seuil = seuil_i
                    best_valeurs = None
                else:
                    best_valeurs = np.unique(X[:, i])
                    best_seuil = None
        if (i_best != -1):  
            nom_best = LNoms[i_best] if len(LNoms) > 0 else ""
            if best_type_num:
                noeud = NoeudNumerique(i_best, nom_best)
                ((left_data, left_class), (right_data, right_class)) = partitionne(X, Y, i_best, best_seuil)
                noeud.ajoute_fils(
                    best_seuil,
                    construit_AD_gen(left_data, left_class, epsilon, LNoms, LTypes, verbose),
                    construit_AD_gen(right_data, right_class, epsilon, LNoms, LTypes, verbose)
                )
            else:
                noeud = NoeudCategoriel(i_best, nom_best)
                for v in best_valeurs:
                    Xv = X[X[:, i_best] == v]
                    Yv = Y[X[:, i_best] == v]
                    fils = construit_AD_gen(Xv, Yv, epsilon, LNoms, LTypes, verbose)
                    noeud.ajoute_fils(v, fils)
        else: 
            noeud = NoeudCategoriel(-1, "Label")
            noeud.ajoute_feuille(classe_majoritaire(Y))
    return noeud

class ClassifierArbreDecision(cl.Classifier):
    def __init__(self, input_dimension, epsilon, LNoms=[]):
        super().__init__(input_dimension)  
        self.__epsilon = epsilon
        self.__LNoms = LNoms
        self.__racine = None
        
    def __str__(self):
        return super().__str__() + ' - ArbreDecision [' + str(super().get_dimension()) + '] eps=' + str(self.__epsilon)
        
    def train(self, desc_set, label_set, verbose=False):
        self.__racine = construit_AD(desc_set, label_set, self.__epsilon, self.__LNoms, verbose)
    
    def score(self, x):
        pass
    
    def predict(self, x):
        return self.__racine.classifie(x)
        
    def accuracy(self, desc_set, label_set):
        total = len(label_set)
        correct = 0
        for i in range(total):
            pred = self.predict(desc_set[i])
            if pred == label_set[i]:
                correct += 1
        return correct / total  
                    
    def number_leaves(self):
        return self.__racine.compte_feuilles()
    
    def draw(self, GTree):
        self.__racine.to_graph(GTree)

class ClassifierArbreNumerique(cl.Classifier):
    def __init__(self, input_dimension, epsilon, LNoms=[]):
        super().__init__(input_dimension)  
        self.__epsilon = epsilon
        self.__LNoms = LNoms
        self.__racine = None

    def __str__(self):
        return super().__str__() + ' - ArbreNumerique [' + str(super().get_dimension()) + '] eps=' + str(self.__epsilon)

    def train(self, desc_set, label_set, verbose=False):
        self.__racine = construit_AD_num(desc_set, label_set, self.__epsilon, self.__LNoms, verbose)

    def score(self, x):
        pass

    def predict(self, x):
        if self.__racine is None:
            return 0
        return self.__racine.classifie(x)

    def accuracy(self, desc_set, label_set):
        total = len(label_set)
        correct = 0
        for i in range(total):
            pred = self.predict(desc_set[i])
            if pred == label_set[i]:
                correct += 1
        return correct / total  

    def number_leaves(self):
        return self.__racine.compte_feuilles()

    def affiche(self, GTree):
        self.__racine.to_graph(GTree)

class ClassifierArbreGeneral(cl.Classifier):
    def __init__(self, input_dimension, epsilon, LNoms=[], LTypes=[]):
        super().__init__(input_dimension)  
        self.__dimension = input_dimension
        self.__epsilon = epsilon
        if len(LNoms) == 0:
            self.__LNoms = ['att_' + str(i) for i in range(0, input_dimension)]
        else:
            self.__LNoms = LNoms
        if len(LTypes) == 0:
            self.__LTypes = [True for _ in range(input_dimension)]
        else:
            self.__LTypes = LTypes
        self.__racine = None

    def __str__(self):
        return super().__str__() + ' - ArbreGeneral [' + str(super().get_dimension()) + '] eps=' + str(self.__epsilon)

    def train(self, desc_set, label_set, verbose=False):
        self.__racine = construit_AD_gen(desc_set, label_set, self.__epsilon, self.__LNoms, self.__LTypes, verbose)

    def score(self, x):
        pass

    def predict(self, x):
        if self.__racine is None:
            return 0
        return self.__racine.classifie(x)

    def accuracy(self, desc_set, label_set):
        total = len(label_set)
        correct = 0
        for i in range(total):
            pred = self.predict(desc_set[i])
            if pred == label_set[i]:
                correct += 1
        return correct / total  

    def number_leaves(self):
        return self.__racine.compte_feuilles()

    def affiche(self, GTree):
        self.__racine.to_graph(GTree)

def tirage(VX, m, avecRemise=False):
    if not avecRemise: 
        return random.sample(VX, m) 
    else: 
        return np.random.choice(VX, size=m, replace=True)


def echantillonLS(LS, m, avecRemise):
    (desc, labels) = LS
    v_indices = np.arange(desc.shape[0])
    tirage_index = tirage(v_indices, m, avecRemise)
    return (desc[tirage_index], labels[tirage_index])


class ClassifierBaggingTree(cl.Classifier):
    def __init__(self, input_dimension, nbArbres, pourc, epsilon, avecRemise):
        super().__init__(input_dimension)  
        self.__nbArbres = nbArbres
        self.__pourc = pourc
        self.__avecRemise = avecRemise
        self.__epsilon = epsilon 
        self.__foret = []    
        self.__m = None      
             
    def __str__(self):
        return super().__str__() + ' - ClassifierBaggingTree eps=' + str(self.__epsilon) + ' nb_arbres=' + str(self.__nbArbres)

    def train(self, desc_set, label_set=None, verbose=False):
        if label_set is None:
            (desc, labels) = desc_set
        else:
            desc, labels = desc_set, label_set

        n_lig, n_col = desc.shape

        self.__m = int(self.__pourc * n_lig)
        if self.__m < 1:
            self.__m = 1  

        self.__foret = []

        for i in range(self.__nbArbres): 
            (d_desc, d_label) = echantillonLS((desc, labels), self.__m, self.__avecRemise)
            arbre = construit_AD_num(d_desc, d_label, self.__epsilon)
            self.__foret.append(arbre)
        
    def add_tree(self, LS):
        (desc, labels) = LS
        n_lig, n_col = desc.shape
        
        if self.__m is None:
            self.__m = int(self.__pourc * n_lig)
            
        (d_desc, d_label) = echantillonLS(LS, self.__m, self.__avecRemise)
        arbre = construit_AD_num(d_desc, d_label, self.__epsilon)
        self.__foret.append(arbre)
        self.__nbArbres += 1 
        
    def score(self, x):
        if len(self.__foret) == 0:
            return 0.0
            
        total_votes = 0
        for arbre in self.__foret:
            vote = arbre.classifie(x)
            if vote is not None:
                total_votes += vote
                
        return total_votes
            
    def predict(self, x):
        total_score = self.score(x)
        if total_score >= 0:
            return 1
        else:
            return -1

class ClassifierBaggingTreeOOB(cl.Classifier):
    def __init__(self, input_dimension, nbArbres, pourc, epsilon, avecRemise):
        super().__init__(input_dimension)  
        self._nbArbres = nbArbres
        self._pourc = pourc
        self._avecRemise = avecRemise
        self._epsilon = epsilon 
        self._foret = []    
        self._m = None
        self._oob_indices = [] 

    def __str__(self):
        return super().__str__() + ' - ClassifierBaggingTreeOOB eps=' + str(self._epsilon) + ' nb_arbres=' + str(self._nbArbres)

    def train(self, desc_set, label_set=None, verbose=False):
        if label_set is None:
            (desc, labels) = desc_set
        else:
            desc, labels = desc_set, label_set

        n_lig, n_col = desc.shape

        self._m = int(self._pourc * n_lig)
        if self._m < 1:
            self._m = 1  

        self._foret = []
        self._oob_indices = []
        all_indices = np.arange(n_lig)

        for i in range(self._nbArbres):
            idx_bootstrap = tirage(list(all_indices), self._m, self._avecRemise)
            idx_bootstrap_set = set(idx_bootstrap)

            idx_oob = np.array([j for j in all_indices if j not in idx_bootstrap_set])

            d_desc  = desc[idx_bootstrap]
            d_label = labels[idx_bootstrap]

            arbre = construit_AD_num(d_desc, d_label, self._epsilon)
            self._foret.append(arbre)
            self._oob_indices.append(idx_oob)

    def oob_score(self, LS):
        """ Calcule le score OOB — estimation de la performance sans jeu de test """
        (desc, labels) = LS
        n_lig = desc.shape[0]
        
        votes      = np.zeros(n_lig)
        nb_votes   = np.zeros(n_lig) 
        
        for arbre, idx_oob in zip(self._foret, self._oob_indices):
            if len(idx_oob) == 0:
                continue
            for i in idx_oob:
                vote = arbre.classifie(desc[i])
                if vote is not None:
                    votes[i]    += vote
                    nb_votes[i] += 1
        
        mask = nb_votes > 0
        # ЗАЩИТА: если ни один объект не получил OOB голосов, возвращаем 0
        if not np.any(mask):
            return 0.0
            
        predictions = np.where(votes[mask] >= 0, 1, -1)
        return np.mean(predictions == labels[mask])

    def add_tree(self, LS):
        (desc, labels) = LS
        n_lig, n_col = desc.shape
        
        if self._m is None:
            self._m = int(self._pourc * n_lig)
            
        all_indices   = np.arange(n_lig)
        idx_bootstrap = tirage(list(all_indices), self._m, self._avecRemise)
        idx_oob       = np.array([i for i in all_indices if i not in set(idx_bootstrap)])
        
        d_desc  = desc[idx_bootstrap]
        d_label = labels[idx_bootstrap]
        
        arbre = construit_AD_num(d_desc, d_label, self._epsilon)
        self._foret.append(arbre)
        self._oob_indices.append(idx_oob)
        self._nbArbres += 1 
        
    def score(self, x):
        if len(self._foret) == 0:
            return 0.0
        total_votes = 0
        for arbre in self._foret:
            vote = arbre.classifie(x)
            if vote is not None:
                total_votes += vote
        return total_votes
            
    def predict(self, x):
        return 1 if self.score(x) >= 0 else -1