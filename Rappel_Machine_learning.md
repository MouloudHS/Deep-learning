# Introduction au DEEP learning

## Definition:

- Le machine learning est un domaine de l'IA qui consiste à programmer une machine pour que celle-ci apprenne à réaliser des tâches en étudiant des exemples de ces dernières.

- D'un point de vue mathématique, ces exemples sont représentés par des données que la machine utilise pour développer un modele.

- Si nous prenons par exemple des données qui semblent être distribue selon la distribution linéaire  `f(x) = à x + b`. Le but du Machine learning serait de trouver les paramètres `à` et `b` qui donnent le meilleur modele possible, et donc qui s'ajuste le mieux à nos données.


- Dans ce cas simple, les meilleurs paramètres `a` et `b` sont ceux qui minimisent le plus la distance entre les données et le modèle.

- D'une autre façon, le ML consiste à développer un modèle en se servant d'un algorithme d'optimisation pour minimiser les erreurs entre les modèle et nos données.


## Modeles de ML:

- On en cite plusieurs, des `modèles linéaires`, des `arbres de décision` et des `supports vector Machines`. Chacun de ses modeles vient avec son propre algorithme d'optimisation:

`modèles linéaires -> Descente de gradient`
`Arbres de decision -> Algorithme CART`
`Support vector machines -> Marge maximum`

- Le Deep learning est un domaine du machine learning dans lequel, au lieu de developper un des modeles ci-dessous, on developpe a la place un reseau de neurones artificiels. 

- Le principe en soit est le meme, on fournit a la machine des donnees et celle ci va se servir d'un algorithme d'optimisation pour ajuster le modele a ces donnnes.

- Cependant, cette fois ci, notre modele n'est pas une simple fonction `f(x) = a*x + b` mais un reseau de fonction connectees les unes aux autres appelle **reseau de neurones**. Plus ces reseaux sont profonds (contient plus de fonctions a l'interieur), plus la machine est capable d'apprendre a realiser des taches complexes (reconnaissance d'objets et de personnes, conduction de voitures etc...).

## Histoire des reseaux de neurones artificiels

- Historiquement, le premier neurone a ete developpee par Warn McCulloc et Walter Pitts en s'inspirant du fonctionnement du neurone biologique. 

### Fonctionnement d'un neurone biologique

![Neurone biologique](neurone.png)

- Les neurones sont des cellules excitables connectées les unes aux autres et ayant pour role de transmettre des informations dans notre systeme nerveux. Ils sont formes de **dendrites**, un **corps cellulaire** et un **axone**. 

- Les dendrites sont les portes d'entrée d'un neurone. A l'entrée d'une dendrite (la synapse), le neurone recoit des signaux lui provenant des neurones qui le precedent. 

- Ces signaux peuvent etre de type éxcitateur ou inhibiteur (c'est comme si on avait des signaux qui valent +1 et d'autres qui valent -1). Lorsque la somme de ces signaux dépasse un certain seuil, le neurone s'active et produit un signal éléctrique. Ce signal circule le long de l'axone en direction des terminaisons pour etre envoyé a son tour vers d'autres neurones de notre systeme nerveux qui fonctionnent de la meme facon biensur.

- Warren et Pitts ont essayé de modéliser ce comportement en supposant qu'un neurone pouvait etre modélisé par une fonction de transfert qui prent en entrée des signaux **X** et qui retourne une sortie **Y**. A l'interieur de cette fonction, il y'a deux grandes étapes. La premiere est une étape d'aggrégation, on fait la somme de toutes les entrées du neurone en multipliant au passage chaque entrée par un coeffiscient W. Ce coeffiscient représente en fait l'activité synaptique:


![Neurone articfiel](neurone-artificiel.png)


- Aggrégation: 

![equation](http://www.sciweavers.org/tex2img.php?eq=f%20%3D%20w_1%20x_1%20%2B%20w_2%20x_2%20%2B%20w_3%20x_3%0A&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

- Activation: 

![equation](http://www.sciweavers.org/tex2img.php?eq=%20%5Cbegin%7Bcases%7Dy%3D1%20%26%20f%20%5Cgeq%200%5C%5Cy%3D0%20%26%20f%20%3C%200%5Cend%7Bcases%7D%20&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

- On note que, dans ce cas, le seuil d'activation est "f = 0".

- Dans le papier de McCulloc et Pitts, on montre que, en reunissant plusieurs neurones artificiels, on peut resoudre n'importe quel probleme le logique booleene. 

- Le probleme avec ce modele proposé par McCulloc et Pitts est qu'il ne dispose pas d'algortihmes d'apprentissage, ce qui le rend incapable a résoudre des problemes venant du monde réel.

### Le perceptron

- En 1957, Franck Rosenblatt (Psychologue américain) trouva comment on peut ameliorer ce modele en introduisant le premier algorithme d'apprentissage de l'histoire du Deep Learning; **le perceptron**.

- Le perceptron est un neurone artificiel qui s'active lorsque la somme pondérée de ses entrées dépasse un certain seuil. Mais en plus de ca, le perceptron dispose aussi d'un algorithme d'apprentissage lui permettant de trouver les valeurs de ses parametres W afin d'obtenir les sorties **Y** qui nous conviennent.

- Perceptron :  `X = {x1, x2} ----[f]---- y` 

- Pour developper le perceptron, Franck s'est inspiré de la théorie de Hebb, celle-ci suggere que lorsque deux neurones biologiques sont excités conjointement, ils renforcent leur lien synaptique. En neurosceinces, c'est ce qu'on appelle la plasticité synaptique et c'est ce qui nous permet de construire sa mémoire, d'apprendre de nouvelles choses ou encore de faire des nouvelles associations.

- A partir de cette idée, Franck a développé un algorithme d'apprentissage qui consiste a entrainer un neurone artificiel sur des données de référence (X,y) pour que celui-ci renforce ces parametres W a chaque fois qu'une entrée **X** est activée en meme temps que la sortie **Y** présente dans ces données. Pour ce faire, il a imaginé la formule ci-dessous, dans laquelle les parametres W sont mis a jour en calculant la différence:

![equation](http://www.sciweavers.org/tex2img.php?eq=W%20%3D%20W%20%2B%20%5Calpha%20%28y_%7Btrue%7D%20-%20y%29X&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

- Avec:
    - `y_{true}` est la sortie de référence.
    - `y` est la sortie produite par le neurone.
    - `X` est l'entrée du neurone.
    - `\alpha` est la vitesse d'apprentissage.

### Le perceptron multicouches

- En 1986, Geoffrey Hinton développa le premier véritable réseau de neurones artificiels; le perceptron multicouches. En effet, le perceptron de Rosenblatt est un modele lineaire. Il est bien convenable si l'on veut par exemple séparer deux classes de points distribués de telle sorte qu'ils soient séparables par une ligne droite, mais pas plus que ca (`f = W.X + b`) avec W une matrice diagonale 2x2 et b un parametre supplémentaire dit le biais. 

![perceptron a deux couches](perceptron-multi.png)

- En construisant un perceptron a deux couches, on obtient un resultat bien plus interessant. On peut maintenant separer deux classes de points distribués non linéairement. Nous enviseagons donc que plus on a de couches et de fonctions de trasnfert, plus notre resultat est assez complexe et plus interessant.

- Cependant, une question subsiste... Comment entrainer un tel réseau de neurones pour qu'il fasse ce qu'on lui demande de faire ? C.a.d. Comment trouver les valeurs de tous les parametres **W** et **b** de facon a ce que l'on obtienne un bon modele.

- La solution proposée par Hinton est d'utiliser une technique appelée Back Propagation qui consiste a déterminer comment la sortie varie en fonction des parametres présents dans chaque couche du modele. Pour cela, on calcul une chaine de gradients, indiquant comment la sortie varie en fonction de la derniere couche de facon recursive.

![perceptron multi-couches](Perceptron_multicouche.png)

- Grace aux gradients, on peut mettre a jour les parametres (W,b) de chaque couche de telle sorte a ce qu'ils minimisent l'erreur entre la sortie du modele et la réponse attendue. Pour cela, on utilise l'algorithme de descente de gradient:

![equation](http://www.sciweavers.org/tex2img.php?eq=W%20%3D%20W%20-%20%5Calpha%20%20%5Cfrac%7B%5Cpartial%20Erreur%7D%7B%5Cpartial%20W%7D%20&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)













