# Défi data Rakuten France  
Le défi consiste à proposer des modèles de Machine Learning capables de réaliser une classification mutli-classes avec des entrées multi-modales (texte et image) permettant de prédire correctement le code produit pour des articles de vente en ligne.  
L'enjeu est d'améliorer le référencement des articles grâce à des modèle de ML qui permettent d'augmenter significativement les scores des modèles de références.  
(https://challengedata.ens.fr/challenges/35)


## Analyse du jeu de données

<img width="600" alt="Data_Set" src="https://github.com/JTh34/data-science/assets/79744432/dcca249e-4a2e-4551-b11e-c3713060c980">

_<sub> Échantillon du jeu du données rassemblé dans un DataFrame</sub>_ 

Le jeu de données comporte 84 916 articles répartis en 27 classes (_'prdtypecode'_).  
À chaque article correspond une image couleur de taille 500 x 500 pixels (accessible via les informations _'productid'_	et _'imageid'_) et des données de texte (_'designation'_	et _'description'_).  

<img width="530" alt="Echantillon_jeu_donnée" src="https://github.com/JTh34/data-science/assets/79744432/de58d740-7dfc-42c9-b1f4-a0acd05381c6">  

_<sub>Échantillon d'images du jeu du données</sub>_

  
<img width="700" alt="Distribution_articles" src="https://github.com/JTh34/data-science/assets/79744432/939a9eb3-afdd-4425-8ecf-0086f21c25b4"> 

 _<sub>Distribution des articles dans chaque catégorie</sub>_ 


## Modèle pour le traitement du texte seul
Pour ce modèle, uniquement les champs de texte des articles sont pris en considération.  
Le champs _'description'_ n'étant pas systématiquement renseigné, seul le champ _'designation'_	est utilisé.  
Un pré-traitmement est effectué pour éliminer les "stop words" ainsi que les éléments de moins de 2 caractères.
Après traitement, le champ  _'designation'_ de chaque article est transformé en une séquence de 33 tokens (longueur maximale).
Pour les articles dont la tokenization renvoie une séquence de longueur inférieure, un remplissage avec des 0 est effectué pour que la taille des séquences d'entrée du modèle soit constante.

Après plusieurs tentativess, le modèle qui semble donner les meilleurs résultats est une architecture de réseau de neuronnes avec une couche d'embedding en entrée.  
Les séquences de tokens sont projettées dans un espace latent dense de dimension 100 où les poids sont initialisés avec ceux d'un modèle Glove 6B 100d. Ces poids sont ajustés ensuite par l'entrainement des poids du réseau de neuronnes avec les données du problème.

<img width="250" alt="model_301" src="https://github.com/JTh34/data-science/assets/79744432/d454a011-1895-4ef0-a777-c670368ae515">  

 _<sub>Architecture du modèle texte</sub>_ 

## Modèle pour le traitement de l'image seule
Pour ce modèle, les photos des articles sont pris en considération.  

Dans un premier temps, un jeu données réduit est utilisé (30% du jeu initial en gardant la même distibution intra-classes) pour sélectionner les modèles les plus intéressants.

<img width="400" alt="Result_ima" src="https://github.com/JTh34/data-science/assets/79744432/84346165-45ea-481b-ac94-23bbf6180b16">  

 _<sub>Performances des modèles sur un jeu de données réduit</sub>_ 

Le modèle retenu est construit sur la base d'un modèle "Xception" pré entraîné avec les 26 dernières couches dégelées.  
Bien que l’accuracy obtebue soit un peu moindre que pour un modèle ayant une base "EfficientNetB5", avec "Xception" il est possible de sauvegarder intégralement le modèle au format .joblib, contrairement au modèle de la série EfficientNet.  
C'est une limitaiton connue intrinsèque à Tensorflow 2+.
Le rappel d’un modèle entraîné au format .joblib est bien plus simple et plus rapide à mettre en œuvre pour les prédictions.



_<sub>Architecture du modèle image</sub>_ 

 _<sub>Performances après un entrainement sur l'intégralité du jeu de données images</sub>_ 
