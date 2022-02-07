# Rapport ML

## Interface graphique

Interface simple avec deux pages différentes : 

- **mainUI** est le point d'entrée de l'interface. Des onglets permettent d'accèder aux différentes pages.
- **pagePredict** qui contient une page permettant de choisir un algorithme parmi KNN, Naïve Bayes et CNN afin de l'executer. Cette page contient un cadre qui permet de dessiner le nombre à reconnaître.
- **pageExperiments** qui permet la sélection d'une expérience et d'en visualiser les différents résultats : accuracy, précision, recall et f-mesure.

L'interface à été réaliser avec le module Tkinter de python.

## Base de données

Nous avons décidé d'utiliser la basee de données mnist pour entraîner et tester nos données.

Dans un premier temps, nous avions créé des images en format .png directement dans dans des dossiers correspondant à leurs classes respectives et nous utilisions Pathlib pour les récupérer.

Comme cela prenait trop de temps, nous avons utilisé la librairie numpy pour stocker les données dans le format .npy. Cela a amélioré nos performances.

Finalement, comme nous voulions stocker aussi les résultats des expériences que nous avions menées, nous avons finalement décidé de créer une base de données en sqLite. Cela nous a permis de garder de bonne performances et d'enregistrer nos données au même endroit.

## Algorithmes

### Naive Bayes

Nous avons choisi d'executer l'algorithme de Naive Bayes sur les données de mnist reformatées : les images sont représentées avec des pixels ne pouvant prendre comme valeur que 0 ou 1. Nous avons divisé la valeur des pixels par 128 pour ramener les valeurs entre 0 et 1.

Nous avons utiliser le module Numpy pour améliorer les performances. Pour calculer la prédiction, nous avons stocké pour chaque classes le nombre de pixel à 1 pour chaque pixels ainsi que le nombre total d'image pour chaque classe. Ainsi nous avions toutes les données nécessaire pour le calcul de Naive Bayes.

### KNN

Dans un premier temps, nous avons utilisé KNN sous le même format de données que pour Naive Bayes en utilisant la distance L2 pour calculer la prédiction. Cela prenait énormément de temps et nous avons donc chercher un moyen d'optimiser.

Nous avons donc décider de formater les données afin de pouvoir utiliser la distance de Levenshtein en utilisant le codage de Freeman. Nous avons implémenté Levenshtein mais nos performance de temps était pire qu'auparavent.

Nous avons donc trouvé une autre solution. Nous avons implémenté l'algorithme de BallTree qui permet d'éviter de parcourir toutes les images dans l'ensemble d'entraînement afin de déterminer les K plus proches voisins.

Afin de ne pas recalculer le ball Tree à chaque prédiction, nous avons stocké l'objet dans un fichier grace au module Pickle de Python.

### CNN

Nous avons récupéré en ligne (https://www.kaggle.com/stephanedc/tutorial-cnn-partie-1-mnist-digits-classification) un modèle de réseau de neurones qui puisse traiter les données sous la forme de tableau de 0 et de 1. 