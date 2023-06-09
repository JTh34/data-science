# Défi Data - Rakuten France  
<div align="justify">
Ce défi data consiste à proposer des modèles de Machine Learning capables de réaliser une classification mutli-classes avec des entrées multi-modales (texte et image) permettant de prédire correctement le code produit pour des articles de vente en ligne.  
L'enjeu est d'améliorer le référencement des articles grâce à des modèle de ML qui permettent d'augmenter significativement les scores des modèles de références (https://challengedata.ens.fr/challenges/35).
</div>


## Analyse du jeu de données

<img width="600" alt="Data_Set" src="https://github.com/JTh34/data-science/assets/79744432/dcca249e-4a2e-4551-b11e-c3713060c980">

_<sub> Échantillon du jeu du données rassemblé dans un DataFrame</sub>_ 

<div align="justify">Le jeu de données comporte 84 916 articles répartis en 27 classes (<i>'prdtypecode'</i>).  
À chaque article correspond une image couleur de taille 500 x 500 pixels (accessible via les informations <i>'productid'</i> et <i>'imageid'</i>i>) et des données de texte (<i>'designation'</i> et <i>'description'</i>).  
</div>

<img width="530" alt="Echantillon_jeu_donnée" src="https://github.com/JTh34/data-science/assets/79744432/de58d740-7dfc-42c9-b1f4-a0acd05381c6">  

_<sub>Échantillon d'images du jeu du données</sub>_

  
<img width="700" alt="Distribution_articles" src="https://github.com/JTh34/data-science/assets/79744432/939a9eb3-afdd-4425-8ecf-0086f21c25b4"> 

 _<sub>Distribution des articles dans chaque catégorie</sub>_ 


## Modèle pour le traitement du texte seul
<div align="justify">
Pour ce modèle, uniquement les champs de texte des articles sont pris en considération.  
Le champs <i>'description'</i> n'étant pas systématiquement renseigné, seul le champ <i>'designation'</i> est utilisé.

Un pré-traitmement est effectué pour éliminer les "stop words" ainsi que les éléments de moins de 2 caractères.
Après traitement, le champ  <i>'designation'</i> de chaque article est transformé en une séquence de 33 tokens (longueur maximale).
Pour les articles dont la tokenization renvoie une séquence de longueur inférieure, un remplissage avec des 0 est effectué pour que la taille des séquences d'entrée du modèle soit constante.

Après plusieurs tentatives, le modèle qui semble donner les meilleurs résultats est une architecture de réseau de neuronnes avec une couche d'embedding en entrée.  
Les séquences de tokens sont projettées dans un espace latent dense de dimension 100 où les poids sont initialisés avec ceux d'un modèle Glove 6B 100d. Ces poids sont ajustés ensuite par l'entrainement des poids du réseau de neuronnes avec les données du problème.
</div>

<img width="250" alt="model_301" src="https://github.com/JTh34/data-science/assets/79744432/d454a011-1895-4ef0-a777-c670368ae515">  

 _<sub>Architecture du modèle texte</sub>_ 

<img width="300" alt="F1_texte" src="https://github.com/JTh34/data-science/assets/79744432/9969efaa-5573-4928-945d-283f337f9b53"> 

<img width="550" alt="Classes_dispatch_texte" src="https://github.com/JTh34/data-science/assets/79744432/75f67265-2b3d-4721-82be-40406aa20318">  

 _<sub>Performances du modèle texte, après entrainement sur l'intégralité du jeu de données</sub>_ 

## Modèle pour le traitement de l'image seule
<div align="justify">
Pour ce modèle, seules les photos des articles sont prises en considération. Dans un premier temps, un jeu de données réduit est utilisé (30% du jeu initial tout en gardant la même distribution intra-classes) pour sélectionner les modèles les plus intéressants.
</div>

<img width="400" alt="Result_ima" src="https://github.com/JTh34/data-science/assets/79744432/84346165-45ea-481b-ac94-23bbf6180b16">  

 _<sub>Performances des modèles sur un jeu de données réduit</sub>_ 

<div align="justify">
Le modèle retenu est construit sur la base d'un modèle "Xception" pré-entraîné dont les 26 dernières couches ont été dégelées.  
Bien que l’accuracy obtenue soit un peu moindre que celle obtenue avec un modèle ayant une base "EfficientNetB5", avec "Xception" il est possible de sauvegarder intégralement le modèle au format <i>.joblib</i>, contrairement au modèle de la série EfficientNet (c'est une limitaiton intrinsèque à Tensorflow 2+).
Le rappel d’un modèle entraîné au format <i>.joblib</i> est bien plus simple et plus rapide à mettre en œuvre pour les prédictions.
</div>

<img width="350" alt="model_51" src="https://github.com/JTh34/data-science/assets/79744432/4e4af3c1-be34-432c-9656-e0187ef58aed"> 

_<sub>Architecture du modèle image</sub>_ 

<img width="300" alt="F1_image_fulldata" src="https://github.com/JTh34/data-science/assets/79744432/19707f3a-4ac7-40e0-b221-7ca0b33eeee5"> 

<img width="550" alt="Classes_dispatch_image" src="https://github.com/JTh34/data-science/assets/79744432/919dfe33-623e-40af-8eee-1fc24977528a">  

 _<sub>Performances du modèle image, après entrainement sur l'intégralité du jeu de données</sub>_ 

## Modèle pour le traitement du texte et de l'image simultanément
<div align="justify">
Pour ce modèle, le principe est de tirer parti des deux meilleurs modèles précédents en concaténant leurs architectures.
Le modèle ainsi obtenu devrait avoir des performances bien meilleures que chaque modèle isolé puisqu'il prend en compte plus de caractéristiques des données d'entrées.
</div>
<img width="450" alt="model_201" src="https://github.com/JTh34/data-science/assets/79744432/e01871de-a878-4e80-88af-df515924da4a"> 

_<sub>Architecture du modèle traitant l'image et le texte simultanément</sub>_ 

<img width="300" alt="F1_texte+image_full_data" src="https://github.com/JTh34/data-science/assets/79744432/880608b4-866c-49c1-934d-29d0691c6a13"> 

<img width="550" alt="Classes_dispatch_texte+image" src="https://github.com/JTh34/data-science/assets/79744432/55813107-e946-42a9-87e8-8cf56e41a913">  

 _<sub>Performances du modèle texte+image, après entrainement sur l'intégralité du jeu de données</sub>_ 


## Analyse des résultats
<img width="650" alt="F1_regroupe_2" src="https://github.com/JTh34/data-science/assets/79744432/cbdb0f00-dbf8-4646-b0b3-c87d815b0ea3"> 

 _<sub>Comparaison des F1-score de chaque modèle pour chaque catégorie</sub>_ 

<div align="justify">
Le modèle qui traite uniquement les images est beaucoup moins performant que ceux qui traitent le texte.  
Le modèle "texte+image" donne globalement de meilleurs résultats que les deux autres.  
Cependant, le modèle qui traite le texte seul est très performant. Au regard de la complexité de la mise en oeuvre et du coût en calculs pour les modèles traitant les images, c'est assez remarquable.
<br></br>
Concernant le défi data, les scores de benchmark ont été notablement dépassés. L'accuracy de référence pour le modèle traitant le texte seul et celle du modèle ne taitant que l'image étaient de **0.81** et **0.55**. Les modèles proposés permettent d'atteindre des accuracy, respectivement, de **0.91** et **0.62**.
</div>

## Quelques prédictions choisies au hasard 

<img width="950" alt="Predictions_Models1" src="https://github.com/JTh34/data-science/assets/79744432/c1760699-efe9-4140-8aa1-c1a6e560b0ab"> 
<!-- -->
<img width="950" alt="Predictions_Models9" src="https://github.com/JTh34/data-science/assets/79744432/506c2144-2788-47b4-8af9-9a578d3913d0"> 

<img width="950" alt="Predictions_Models3" src="https://github.com/JTh34/data-science/assets/79744432/45f1cdf4-1cb7-44f6-8701-0101ad73514a"> 

<img width="950" alt="Predictions_Models11" src="https://github.com/JTh34/data-science/assets/79744432/df57ba62-4089-48b2-8de6-217f4b075ccd"> 

<img width="950" alt="Predictions_Models5" src="https://github.com/JTh34/data-science/assets/79744432/551218cb-0582-429d-b6e8-372e74f69fd1"> 

