import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

FASHION_LABELS = {
    0: "T-shirt/top",
    1: "Pantalon",
    2: "Pull",
    3: "Robe",
    4: "Manteau",
    5: "Sandale",
    6: "Chemise",
    7: "Basket",
    8: "Sac",
    9: "Bottine"
}

def plot_all_fashion_2d(X, Y, title="Distribution globale de Fashion MNIST (PCA 2D)"):
    """
    Projette l'intégralité des 10 classes en 2D via PCA et les affiche sur un scatter plot.
    """
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    plt.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray', alpha=0.5)
    
    unique_classes = np.unique(Y)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
    
    for idx, c in enumerate(unique_classes):
        mask = (Y == c)
        plt.scatter(
            X_2d[mask, 0], 
            X_2d[mask, 1], 
            color=colors[idx], 
            label=FASHION_LABELS[int(c)], 
            alpha=0.6, 
            edgecolors='none', 
            s=25
        )
    
    plt.xlabel('x1 (Première Composante Principale)')
    plt.ylabel('x2 (Deuxième Composante Principale)')
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    
    plt.legend(title="Catégories d'habits", loc='upper right', bbox_to_anchor=(1.25, 1), frameon=True)
    
    plt.tight_layout()
    plt.show()


def plot2DTrainTestSet(d_train,l_train, d_test,l_test, nom_dataset= "Dataset", avec_grid=True):    
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
    plt.scatter(desc_neg_test[:,0], desc_neg_test[:,1], marker='o', color="yellow", label="classe -1 test", alpha=0.5, s=30)
    plt.scatter(desc_pos_test[:,0], desc_pos_test[:,1], marker='x', color="green", label="classe +1 test", alpha=0.5, s=30) 
    plt.scatter(desc_neg_train[:,0], desc_neg_train[:,1], marker='o', color="red", label="classe -1", alpha=0.7, s=35)
    plt.scatter(desc_pos_train[:,0], desc_pos_train[:,1], marker='x', color="blue", label="classe +1", alpha=0.7, s=35) 
    plt.title(nom_dataset)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(avec_grid)
    plt.show()

def plot_comparaison_classifiers(models_names, means, stds, title="Comparaison des Classifiers"):
    """ list[str] * list[float] * list[float] * str -> None
        Affiche un diagramme en barres comparatif avec les écarts-types (les "moustaches").
    """
    
    plt.figure(figsize=(9, 6))
    
    bars = plt.bar(models_names, means, yerr=stds, 
                   color=['darkorange', 'cornflowerblue', 'royalblue', 'seagreen', 'crimson'][:len(models_names)], 
                   capsize=8, alpha=0.85, edgecolor='black', width=0.4,
                   error_kw={'ecolor': 'crimson', 'elinewidth': 2, 'capthick': 2})
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.4f}', 
                 ha='center', va='bottom', fontweight='bold', fontsize=10)
                 
    plt.title(title, fontsize=13, fontweight='bold', pad=15)
    plt.ylabel("Accuracy Moyenne (Validation Croisée)", fontsize=11)
    plt.ylim(0.5, 1.0)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def visualiser_frontiere_2d(X, Y, classifier, title="Frontière de Décision", step=50):
    """ array * array * Classifier * str * int -> NoneType
        Prend un dataset 2D, entraîne/utilise le classifieur et affiche 
        la frontière de décision superposée avec les points de données.
    """
    mmax = X.max(0)
    mmin = X.min(0)
    
    x1_min, x1_max = mmin[0] - 0.1, mmax[0] + 0.1
    x2_min, x2_max = mmin[1] - 0.1, mmax[1] + 0.1
    
    x1grid, x2grid = np.meshgrid(np.linspace(x1_min, x1_max, step), 
                                 np.linspace(x2_min, x2_max, step))
    grid = np.hstack((x1grid.reshape(x1grid.size, 1), x2grid.reshape(x2grid.size, 1)))
    
    res = np.array([classifier.predict(grid[i, :]) for i in range(len(grid))])
    res = res.reshape(x1grid.shape)
    
    plt.contourf(x1grid, x2grid, res, colors=["darksalmon", "skyblue"], levels=[-1000, 0, 1000], alpha=0.6)
    
    plt.scatter(X[Y == -1][:, 0], X[Y == -1][:, 1], color='crimson', label='T-shirt (-1)', alpha=0.7, edgecolors='k')
    plt.scatter(X[Y == 1][:, 0], X[Y == 1][:, 1], color='darkblue', label='Chemise (1)', alpha=0.7, edgecolors='k')
    
    plt.title(title, fontsize=11, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='upper right')