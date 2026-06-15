# Importation des bibliothèque
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.decomposition import PCA
from sklearn.metrics import (precision_recall_curve, auc, confusion_matrix,ConfusionMatrixDisplay,
accuracy_score, adjusted_rand_score, normalized_mutual_info_score)
from sklearn.pipeline import Pipeline





def charger_descripteurs(methode):
    '''
    Cette foction:
    prend une méthode en paramètre
    utilise la bibliothèque os pour lister toutes les images dans la base BDshape
    ignore les fichiers ne commençant pas par "s" et ne terminant pas par l'extension de la méthode
    extraire la classe 
    retourne le tableau des featurs et un tableau des classes

    Charge les descripteurs à partir des fichiers dans le dossier ./BDshape/ en fonction de la méthode spécifiée.       
    Args:
        methode (str): La méthode de descripteur à charger 
    '''
    X = []
    y = []
    chemin_dossier="./BDshape/"
    for fichier in sorted(os.listdir(chemin_dossier)):
        if not fichier.lower().endswith(methode.lower()):
            continue
        if not fichier.lower().startswith("s"):
            continue
        
        # Normalisation du nom : on met tout en majuscules
        fichier_upper = fichier.upper()
        
        try:
            classe = int(fichier_upper[1:3])  # Sxx
        except ValueError:
            continue  # Ignore les fichiers non conformes

        chemin_fichier = os.path.join(chemin_dossier, fichier)
        with open(chemin_fichier, "r") as f:
            contenu = f.read().strip()
            valeurs = np.fromstring(contenu, sep=" ")
            X.append(valeurs)
            y.append(classe)
    return np.array(X), np.array(y)



def visualiser_donnees(X, y, methode):
    '''
    Visualise les données en 2D après normalisation et réduction de dimensionnalité avec PCA.
    '''
       
    # Normalisation
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    # ACP
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_norm)

    # Visualisation 
    plt.figure(figsize=(4,3))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='tab10', s=50)
    plt.colorbar(label="Classe")    
    plt.xlabel("Composante principale 1")
    plt.ylabel("Composante principale 2")
    plt.title(f"Visualisation des données en 2D pour {methode}")
    plt.show()
    
    

def rechercher_k_optimal_voisins(X, y, methode):
    '''
    Recherche le k optimal pour KNN en utilisant la validation croisée.
    Affiche un graphique des accuracies en fonction de k.
    Etape:
    - Normalisation des données
    - Boucle sur les k de 1 à 10
    - Pour chaque k, création d'un pipeline normalisation + KNN 
    - Validation croisée 10-fold pour évaluer l'accuracy
    - Stockage des accuracies
    - Tracé du graphique avec le meilleur k mis en évidence
    return le k optimal
    '''

    # Normalisation
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    meilleurs_scores = []
    meilleur_k = None
    meilleur_score = -1

    k_values = range(1, 10)  # k de 1 à 10

    # Boucle sur les k possibles
    for k in k_values:
        
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k))
        ])  
        # Validation croisée
        scores = cross_val_score(model, X_norm, y, cv=10, scoring='accuracy')     
        acc = scores.mean()
        meilleurs_scores.append(acc)

        # Mise à jour du meilleur modèle
        if acc > meilleur_score:
            meilleur_score = acc
            meilleur_k = k

    # --- Tracé du graphique ---
    plt.figure(figsize=(3,2))
    plt.plot(k_values, meilleurs_scores, marker='o', color='blue', label="Accuracy")

    # Trait vertical rouge sur le meilleur k
    plt.axvline(x=meilleur_k, color='red', linestyle='--', label=f"Meilleur k={meilleur_k}")

    # Point rouge au maximum
    plt.scatter([meilleur_k], [meilleur_score], color='red', s=80)

    plt.xlabel("Nombre de voisins (k)")
    plt.ylabel("Accuracy moyenne (CV=10)")
    plt.title(f"k optimal pour {methode} ")
    plt.grid(True)
    plt.legend()
    plt.show()

    return meilleur_k

def approche_knn(descripteurs=["E34", "GFD", "SA", "F0", "F2"], k_optimal=1):
    
    '''
    Approche KNN avec validation croisée et visualisation des résultats.
    Pour chaque méthode de descripteur:
    - Chargement des descripteurs
    - Visualisation des données en 2D
    - Création d'un pipeline normalisation + KNN avec k optimal
    - Validation croisée stratifiée 10-fold
    - Calcul de l'accuracy globale
    - Affichage de la matrice de confusion
    - Tracé des courbes de précision-rappel pour chaque classe avec AUC
    les k_optimaux sont tous egaux à 1 dans notre cas d'après la recherche précédente.
    '''
    accuracies = {} # Dictionnaire pour stocker les accuracies
    
    for methode in descripteurs:
        X, y = charger_descripteurs(methode)
        # -- 1. Visualisation des données --
        visualiser_donnees(X, y, methode)
        
        # --- 1. Pipeline normalisation + KNN ---
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k_optimal))
        ])

        # --- 2. Validation croisée stratifiée ---
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

        # cross_val_predict donne les prédictions "out-of-fold"
        y_pred = cross_val_predict(model, X, y, cv=skf)

        # accuracy global
        acc = accuracy_score(y, y_pred)
        accuracies[methode] = acc
        # --- 3. Matrice de confusion globale ---
        cm = confusion_matrix(y, y_pred)

        # --- FIGURE 1 : CM + Courbes PR ---
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        
        # ===== 1) MATRICE DE CONFUSION =====
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(cmap="Blues", ax=axes[0], colorbar=False)   # IMPORTANT : ax=axes[0]
        axes[0].set_title(f"Confusion Matrix (k={k_optimal}) pour {methode}", fontsize=8)
        axes[0].tick_params(labelsize=6)
    
        # ===== 2) COURBES PR =====
        y_proba = cross_val_predict(model, X, y, cv=skf, method="predict_proba")
        classes = np.unique(y)
        y_bin = label_binarize(y, classes=classes)
        
        auc_scores = []
        for i, c in enumerate(classes):
            precision, recall, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
            auc_pr = auc(recall, precision) 
            auc_scores.append(auc_pr)
            axes[1].plot(recall, precision, label=f"{c} (AUC={auc_pr:.3f})")
        
        axes[1].set_xlabel("Rappel")
        axes[1].set_ylabel("Précision")
        axes[1].set_title(f"Courbes PR – CV 10 Fold pour {methode}", fontsize=8)
        axes[1].legend(fontsize=6)
        axes[1].grid(True)
    
        plt.tight_layout()
        plt.show()
    # Affichage des accuracies
    print("Accuracies pour toutes les classes:")
    for methode, acc in accuracies.items():
        print(f"{methode} : {acc:.3f}")





def recherche_k_optimal_clusters(X, methode):
    '''
    Recherche le k optimal pour K-Means en utilisant la méthode du coude.
    Affiche un graphique du WCSS en fonction de k.
    Etape:
    - Normalisation des données 
    - Boucle sur les k de 1 à 9
    - Pour chaque k, création et entraînement du modèle K-Means
    - Stockage du WCSS (inertie)
    - Tracé du graphique du WCSS en fonction de k
    - Le k est à choisir en fonction du coude visible sur le graphique
    '''
    # Normalisation des données
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    wcss = []  # liste des inerties

    # Calcul du WCSS pour chaque K
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X_norm)
        wcss.append(kmeans.inertia_)

    # Visualisation
    plt.figure(figsize=(3, 2))
    plt.plot(range(1, 10), wcss, marker='o')
    plt.xlabel("Nombre de clusters (k)")
    plt.ylabel("WCSS (inertie)")
    plt.title(f"Méthode du coude pour {methode}")
    plt.grid(True)
    plt.show()
    
  
def approche_kmeans(descripteurs={"E34" : 2, "GFD" : 3, "SA" : 4, "F0" : 3, "F2" : 3}):
    
    '''
    Approche K-Means avec évaluation des clusters.
    Pour chaque méthode de descripteur:
    - Chargement des descripteurs
    - Normalisation des données
    - Entraînement du modèle K-Means avec le k optimal
    - Calcul des indices d'évaluation ARI et NMI
    - Affichage des résultats
    -Les k optimals sont fournis dans le dictionnaire descripteurs. Ils ont été déterminés par la méthode du coude.
    '''
    resultats = {} # Pour stocker les résultats
    for methode, k_optimal in descripteurs.items():
        X, y = charger_descripteurs(methode)
        
        # ---1 Normalisation ---
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X)

        # ---2 Entraînement du modèle K-Means ---
        kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_norm)

        # ---3 Indices d'évaluation ---
        ari = adjusted_rand_score(y, clusters)     # ARI
        nmi = normalized_mutual_info_score(y, clusters)  # NMI
        resultats[methode] = (ari, nmi)
        
    # ---4 Affichage des résultats ---
    for methode, (ari, nmi) in resultats.items():
        print(f"\n===== Résultats K-Means pour {methode} =====")
        print(f"ari: {ari:.2f}\nnmi: {nmi:.2f}")

    
    
def vote_majoritaire_descripteurs(descripteurs=["E34", "GFD", "SA", "F0", "F2"], k_optimal=1):
    
    '''
    Approche de vote majoritaire entre plusieurs descripteurs avec KNN.
    Etapes:
    - Pour chaque descripteur:
        - Chargement des descripteurs
        - Création d'un pipeline normalisation + KNN avec k optimal
        - Prédictions cross-validation (out-of-fold)
        - Calcul de l'accuracy individuelle
    - Vote majoritaire sur les prédictions de tous les descripteurs
    - Calcul de l'accuracy finale après vote
    - Affichage des résultats
    '''
    resultat_descripteurs = {} # Pour stocker l'accuracy par descripteur
    predictions_list = [] # Pour stocker les prédictions

    # --- Validation croisée stratifiée ---
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    # --- 1. Pour chaque descripteur ---
    for methode in descripteurs:
        # Chargement
        X, y = charger_descripteurs(methode)

        # Pipeline : normalisation + KNN
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k_optimal))
        ])

        # Prédictions cross-validation (out-of-fold)
        y_pred = cross_val_predict(model, X, y, cv=skf)

        # Accuracy individuelle
        acc = accuracy_score(y, y_pred)
        resultat_descripteurs[methode] = acc

        # Stockage pour vote
        predictions_list.append(y_pred)

    # --- 2. Vote majoritaire ---
    predictions_array = np.array(predictions_list)  

    # vote par colonne
    y_pred_final = np.apply_along_axis(lambda col: np.bincount(col).argmax(), axis=0, arr=predictions_array)

    # Accuracy vote majoritaire
    acc_final = accuracy_score(y, y_pred_final)

    # --- 3. Affichage ---
    print("Accuracy pour toutes les classes :")
    for methode, acc in resultat_descripteurs.items():
        print(f"{methode} : {acc:.3f}")

    print(f"\nAccuracy vote majoritaire : {acc_final:.3f}")




def vote_majoritaire_descripteurs_5_classes(descripteurs=["E34", "GFD", "SA", "F0", "F2"], k_optimal=1):
    
    '''
    Approche de vote majoritaire entre plusieurs descripteurs avec KNN, limitée aux 5 premières classes.
    Etapes:
    - Pour chaque descripteur:
        - Chargement des descripteurs
        - Filtrage pour ne garder que les classes 1 à 5
        - Création d'un pipeline normalisation + KNN avec k optimal
        - Prédictions cross-validation (out-of-fold)
        - Calcul de l'accuracy individuelle
    - Vote majoritaire sur les prédictions de tous les descripteurs
    - Calcul de l'accuracy finale après vote
    - Affichage des résultats
    '''
    resultats = {} # Pour stocker l'accuracy par descripteur
    predictions = [] # Pour stocker les prédictions
    
    #---1 deschargerement des descripteurs ---
    for methode in descripteurs:
        X, y = charger_descripteurs(methode)
        # classes à garder
        classes_5 = [1, 2, 3, 4, 5]
        # filtrage
        mask = np.isin(y, classes_5)
        X5 = X[mask]
        y5 = y[mask]
        
        #---Pipeline de normalisation + KNN ---
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k_optimal))
        ])
        
        #---Validation croisée stratifiée ---
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
        
        # cross_val_predict donne les prédictions "out-of-fold"
        y_pred = cross_val_predict(model, X5, y5, cv=skf)
        
        # accuracy global/Taux de reconnaissance
        acc = accuracy_score(y5, y_pred)
        resultats[methode] = acc   
        # stockage des prédictions pour le vote
        predictions.append(y_pred)
    #---4 Vote majoritaire ---
    predictions = np.array(predictions)
    y_pred_final = np.apply_along_axis(lambda x: np.bincount(x).argmax(), 0, predictions)
    acc_final = accuracy_score(y5, y_pred_final)
    #---5 Résultats ---
    print("Accuracy par descripteur(5 premiers classes) :")
    for methode, acc in resultats.items():
        print(f"{methode} : {acc:.2f}")
    print(f"Accuracy vote majoritaire : {acc_final:.2f}")




if __name__ == "__main__":
    '''
    Programme principal pour exécuter les différentes approches.
    Pour l'approche KNN:
    - Recherche du k optimal pour chaque descripteur   
    - Exécution de l'approche KNN avec les k optimaux
    Pour l'approche K-Means:
    - Recherche du k optimal pour chaque descripteur
    - Exécution de l'approche K-Means avec les k optimaux
    Pour le vote majoritaire:
    - Vote majoritaire entre descripteurs avec KNN
    - Vote majoritaire entre descripteurs avec KNN (5 premières classes)
    '''
    # Recherche du k optimal pour KNN
    descripteurs_knn = ["E34", "GFD", "SA", "F0", "F2"]
    for methode in descripteurs_knn:
        X, y = charger_descripteurs(methode)
        print(f"Recherche du k optimal pour KNN avec {methode}")
        k_optimal = rechercher_k_optimal_voisins(X, y, methode)
        print(f"k optimal pour {methode} : {k_optimal}\n")
    
    # Approche KNN
    print("=== Approche KNN ===")
    approche_knn(descripteurs=descripteurs_knn, k_optimal=1)
    
    # Recherche du k optimal pour K-Means
    descripteurs_kmeans = ["E34", "GFD", "SA", "F0", "F2"]
    for methode in descripteurs_kmeans:
        X, y = charger_descripteurs(methode)
        print(f"Recherche du k optimal pour K-Means avec {methode}")
        recherche_k_optimal_clusters(X, methode)
    
    # Approche K-Means
    print("=== Approche K-Means ===")
    approche_kmeans(descripteurs={"E34" : 2, "GFD" : 3, "SA" : 4, "F0" : 3, "F2" : 3})
    
    # Vote majoritaire entre descripteurs avec KNN
    print("=== Vote majoritaire entre descripteurs avec KNN ===")
    vote_majoritaire_descripteurs(descripteurs=descripteurs_knn, k_optimal=1)
    
    # Vote majoritaire entre descripteurs avec KNN (5 premières classes)
    print("=== Vote majoritaire entre descripteurs avec KNN (5 premières classes) ===")
    vote_majoritaire_descripteurs_5_classes(descripteurs=descripteurs_knn, k_optimal=1)
