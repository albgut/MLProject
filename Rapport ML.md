# Rapport ML

Suite à une mauvaise manipulation de git, j'ai malencontreusement supprimé le contenu de ma base de données. Ainsi il manque les données concernant KNN. J'ai enregistré un fichier texte au cours du développement qui contient moins d'informations que dans le sql mais qui permet de montrer certains résultats. Vous trouverez l'accuracy et la précision pour chaque digit pour des valeurs de k de 2, 4, 5, 6 et 7. D'ici la présentation, je repeuplerai la base de données.

## Interface graphique

Interface simple avec deux pages différentes : 

- **mainUI** est le point d'entrée de l'interface. Des onglets permettent d'accéder aux différentes pages.
- **pagePredict** qui contient une page permettant de choisir un algorithme parmi KNN, Naïve Bayes et CNN afin de l'exécuter. Cette page contient un cadre qui permet de dessiner le nombre à reconnaître.
- **pageExperiments** qui permet la sélection d'une expérience et d'en visualiser les différents résultats : accuracy, précision, recall et f-mesure.

L'interface à été réalisée avec le module Tkinter de python.

## Base de données

Nous avons décidé d'utiliser la base de données mnist pour entraîner et tester nos données.

Dans un premier temps, nous avions créé des images en format .png directement dans dans des dossiers correspondant à leurs classes respectives et nous utilisions Pathlib pour les récupérer.

Comme cela prenait trop de temps, nous avons utilisé la librairie numpy pour stocker les données dans le format .npy. Cela a amélioré nos performances.

Finalement, comme nous voulions stocker aussi les résultats des expériences que nous avions menées, nous avons finalement décidé de créer une base de données en sqLite. Cela nous a permis de garder de bonnes performances et d'enregistrer nos données au même endroit.

La structure de la base de données est comme suit :

- Table *train* : format (id int, image blob, label text). Contient toutes les images d'entraînement de mnist associées avec leurs labels respectifs (environ 60000 images).
- Table *test* : format (id int, image blob, label text). Contient toutes les images de test de mnist associées avec leurs labels respectifs (environ 10000 images).
- Table *algo* : format (id int, name text). Contient les différents algorithmes implémentés, c'est-à-dire 'Naive Bayes', 'KNN' et 'CNN'.
- Table *exp* : format (id int, date datetime, idAlgo int, timeExec int, accuracy float, k int, distance text). Contient une expérience, c'est à dire le lancement d'un algorithme, avec la date du lancement, l'id de l'algorithme lancé, le temps d'exécution arrondi en secondes, l'accuracy et les hyperparamètres k et distance. Les hyperparamètres non utilisés sont à None.
- Table *predictions* : format (idExp int, idTest int, prediction text). Contient toutes les prédictions associées à une expérience et à une image de test. Le champ prediction contient la prédiction effectuée par l'algorithme.

## Algorithmes

### Naive Bayes

Nous avons choisi d'exécuter l'algorithme de Naive Bayes sur les données de mnist reformatées : les images sont représentées avec des pixels ne pouvant prendre comme valeur que 0 ou 1. Nous avons divisé la valeur des pixels par 128 pour ramener les valeurs entre 0 et 1.

Nous avons utilisé le module Numpy pour améliorer les performances. Pour calculer la prédiction, nous avons stocké pour chaque classes le nombre de pixel à 1 pour chaque pixels ainsi que le nombre total d'images pour chaque classe. Ainsi nous avions toutes les données nécessaire pour le calcul de Naive Bayes.

### KNN

Dans un premier temps, nous avons utilisé KNN sous le même format de données que pour Naive Bayes en utilisant la distance L2 pour calculer la prédiction. Cela prenait énormément de temps et nous avons donc cherché un moyen d'optimiser.

Nous avons donc décider de formater les données afin de pouvoir utiliser la distance de Levenshtein en utilisant le codage de Freeman. Nous avons implémenté Levenshtein mais nos performance de temps était pire qu'qu'auparavant.

Nous avons donc trouvé une autre solution. Nous avons implémenté l'algorithme de BallTree qui permet d'éviter de parcourir toutes les images dans l'ensemble d'entraînement afin de déterminer les K plus proches voisins.

Afin de ne pas recalculer le ball Tree à chaque prédiction, nous avons stocké l'objet dans un fichier grâce au module Pickle de Python.

### CNN

Nous avons récupéré en ligne (https://www.kaggle.com/stephanedc/tutorial-cnn-partie-1-mnist-digits-classification) un modèle de réseau de neurones qui puisse traiter les données sous la forme de tableau de 0 et de 1. 
