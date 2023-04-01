# ENSAE 1A : projet de programmation

Ce dépôt public contient les ressources sur l'optimisation d'un réseau de livraison. 


Ce dépôt contient plusieurs dossiers et fichiers : 
- le dossier `delivery_network` contient le code principal. C'est là qu'est la classe Graph que vous devez implémenter. C'est aussi là que vous mettrez les autres fichiers .py principaux au cours du projet. 
- le dossier 'inputs' contient des jeux de données (graphes et ensembles de trajets) 
- le dossier 'tests' contient les tests unitaires (des exemples, à vous d'en faire d'autres !)
- le fichier `install_graphviz.sh` permet d'installer graphviz sur sspcloud

## Format des fichiers d'input

Le dossier input contient 2 types de fichiers : les fichiers network.x.in ($x \in \{00, 01, 02, 03, 04, 1, ..., 10\}$) qui contiennent les graphes et les fichiers routes.x.in ($x \in \{1, ..., 10\}$) qui contiennent des ensembles de trajets pour les graphes de $x$ correspondant. 

La structure des fichiers network.x.in est la suivante : 
- la première ligne est composée de deux entiers séparés par un espace : le nombre de sommets (n) et le nombre d'arêtes (m)
- les m lignes suivantes représentent chacune une arête et sont composées de 3 ou 4 nombres séparés par des espaces : `ville1 ville2 puissance [distance]`, où `ville1` et `ville2` sont les sommets de l'arête, puissance est la puissance minimale requise pour passer sur l'arête, et distance (optionnel) est la distance entre `ville1` et `ville2` sur l'arête. 

La structure des fichiers routes.x.in est la suivante : 
- la première ligne contient un entier qui correspond aux nombres de trajets dans l'ensemble (T)
- les T lignes suivantes contiennent chacune un trajet sous la forme `ville1 ville2 utilité`, où utilité est le profit acquis si le trajet correspondant est couvert. 
