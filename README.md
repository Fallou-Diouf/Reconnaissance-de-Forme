# BDshape - Reconnaissance des Formes par Classification

## Contexte du Projet

**Projet universitaire** - M1 Master Informatique, année 2025-2026  
**Objectif** : Évaluer le comportement de 5 méthodes classiques de descripteurs de formes 
en les associant à différents classifieurs sur une base complexe (9 classes, 11 échantillons).

### Pourquoi ce projet est pertinent ?
La base BDshape est intentionnellement **complexe** :
- Formes occultées (classes 4, 5)
- Représentations partielles (classes 2, 3, 5)
- Distorsions (classe 6)
- Hétérogénéité (classe 8)

**Enjeu** : Comparer l'efficacité de 5 descripteurs de formes (E34, GFD, SA, F0, F2) 
avec 3 approches de classification différentes.

---

## Mon Rôle dans le Projet

J'ai implémenté et optimisé **l'intégralité de la pipeline de classification** :

✅ **Approche KNN supervisée** : Recherche du k optimal par validation croisée 10-fold + courbes précision-rappel  
✅ **Approche K-Means non supervisée** : Méthode du coude pour déterminer k optimal + indices ARI/NMI  
✅ **Approche Vote Majoritaire** : Fusion des prédictions de 5 descripteurs avec KNN  
✅ **Analyse comparative** : Tests sur 5 classes vs 9 classes pour étudier l'impact de la complexité  

Ma contribution clé : **Architecture modularisée** (5 fonctions principales réutilisables) 
+ **Automatisation de la recherche d'hyperparamètres** + **Visualisations comparatives** (matrices de confusion, courbes PR, AUC).

---

## Technologies & Stack Technique

- **Langage** : Python
- **Descripteurs de formes** : E34, GFD, SA, F0, F2
- **Librairies** :
  - `scikit-learn` : KNeighborsClassifier, KMeans, Pipeline, cross_val_predict, StratifiedKFold
  - `numpy` : Manipulations de données (votes majoritaires, indexation)
  - `matplotlib` : Visualisations (confusion matrices, courbes PR, WCSS)
  - `scikit-learn.preprocessing` : StandardScaler, label_binarize
  - `scikit-learn.decomposition` : PCA (visualisation en 2D)
  - `scikit-learn.metrics` : ARI, NMI, AUC, precision_recall_curve

---

## Résultats Clés

| Approche | Métrique | Résultat |
|----------|----------|----------|
| **KNN (k=1)** | Accuracy moyenne | 89 |
| **K-Means** | ARI moyen | 80 |
| **Vote Majoritaire** | Accuracy (9 classes) | 91 |
| **Vote Majoritaire** | Accuracy (5 classes) | 100 |

## Fonctionnalités Principales

### 1️⃣ Chargement & Normalisation
```python
charger_descripteurs(methode)  # Lit les 5 descripteurs depuis ./BDshape/
Lecture automatique des fichiers .E34, .GFD, .SA, .F0, .F2
Extraction de la classe depuis le nom du fichier (SxxNyyy)
Retour : features (N×D) + labels (N,)
2️⃣ Approche KNN Supervisée
rechercher_k_optimal_voisins(X, y, methode)  # Teste k=1..10
approche_knn(descripteurs, k_optimal=1)      # CV 10-fold + PR curves
Validation croisée stratifiée 10-fold
Matrice de confusion globale
Courbes précision-rappel par classe + AUC
3️⃣ Approche K-Means Non Supervisée
recherche_k_optimal_clusters(X, methode)  # Méthode du coude
approche_kmeans(descripteurs)              # ARI + NMI
Détermination automatique de k via WCSS (inertie)
Indices d'évaluation : ARI (Adjusted Rand Index), NMI (Normalized Mutual Information)
4️⃣ Vote Majoritaire
vote_majoritaire_descripteurs(descripteurs, k_optimal=1)        # 9 classes
vote_majoritaire_descripteurs_5_classes(descripteurs, k_optimal=1)  # 5 classes
Fusion des prédictions de 5 descripteurs
Comparaison impact : complexité base complète vs 5 classes
```

📁 Structure du Projet
BDshape/
├── README.md                 # Ce fichier
├── BDshape/                  # Base de données
│   ├── S01N001.E34, .GFD, .SA, .F0, .F2  # Classe 1, échantillon 1
│   ├── S01N002.E34, ...
│   └── ... (9×11 formes)
├── main.py                   # Script principal (exécution complète)
├── classification.py         # Fonctions réutilisables
└── results/                  # Résultats générés
    ├── confusion_matrices/
    ├── precision_recall_curves/
    └── summary.txt

Comment Exécuter

Installation
pip install numpy scikit-learn matplotlib
Exécution complète
python BDshape_Reconnaissance_des_Formes_par_Classification.py
Cela lance automatiquement :

Recherche des k optimaux pour KNN et K-Means
Approche KNN avec validation croisée
Approche K-Means avec évaluation des clusters
Vote majoritaire (9 classes + 5 classes)
