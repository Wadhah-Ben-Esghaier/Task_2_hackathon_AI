# Méthode 1.2 — choisir les prompts et tester FLIPS sur neuf références

**Question expérimentale.** À budget identique, les prompts sélectionnés sur les déploiements connus améliorent-ils le regroupement par modèle exact ? Vingt réponses sous forme de suites binaires apportent-elles davantage que vingt réponses textuelles supplémentaires ? Le dossier travaille sur `ref-01` à `ref-09`. Les CSV produits servent au score local. Les neuf références appartiennent à la même famille P.

## Exécuter l’expérience

Depuis le dépôt `Task_2_hackathon_AI`, dans un environnement Python contenant `numpy`, `scipy` et `requests` (les dépendances figurent dans le `requirements.txt` du dépôt) :

```sh
cd "méthode 1.2"
python collect.py calibration
python experiment.py fit --answers runs/reference_answers.jsonl
python collect.py test
python experiment.py test --answers runs/reference_answers.jsonl
python experiment.py loo --answers runs/reference_answers.jsonl
```

La première commande de collecte remplit les lots de sélection et de calibration. Le plan figé, les identifiants retenus, les marges et les seuils sont enregistrés dans `runs/plan.json`. La seconde collecte complète le même fichier. On peut relancer une collecte interrompue. Pour réutiliser un JSONL de méthode 1, **copier** ce fichier vers `runs/reference_answers.jsonl` avant la première collecte. Les identifiants, textes et numéros de répétition doivent correspondre. Une collecte peut aussi utiliser `--answers chemin/vers/copie.jsonl`. La clé définie dans la variable d’environnement `API_KEY` a priorité sur la configuration du client existant. Exécuter la collecte sur la machine qui a accès au serveur.

| Lot | Répétitions texte (numérotées à partir de 0) | Répétitions FLIPS | Utilisation |
|---|---|---|---|
| Sélection | 0 à 3, pour 50 prompts | Aucune | Choisir les questions |
| Calibration | 4 à 7, pour 50 prompts | 0 à 4, pour 4 alphabets | Ajuster les échelles et le seuil de chaque méthode |
| Test réservé | 8 à 11, pour 50 prompts | 5 à 9, pour 4 alphabets | Évaluer des décisions figées |

Ces **lots de référence mutualisés** coûtent au total 640 réponses par référence, soit 5 760 générations pour les neuf, si on part de zéro. Ils permettent de rejouer tous les protocoles sur des réalisations comparables. **Chaque protocole évalué ne voit que 80 ou 100 réponses par référence du lot de test.** Les répétitions supplémentaires collectées pour les autres protocoles restent exclues de son calcul.

## Comparaisons

| Nom dans le rapport | Texte par endpoint | Suites binaires | Question |
|---|---:|---:|---|
| `m1_100` | 50 prompts × 2 | 0 | Méthode 1 d’origine à budget 100 |
| `random_100` | 25 × 4 | 0 | Effet du passage à quatre répétitions |
| `margin_100` | 25 × 4 | 0 | Gain de sélection face à `random_100` |
| `random_80` | 20 × 4 | 0 | Témoin aléatoire à 80 réponses |
| `margin_80` | 20 × 4 | 0 | Sélection à 80 et témoin du modèle hybride |
| `margin_flips_100` | Les mêmes 20 × 4 | 4 alphabets × 5 | FLIPS ou cinq prompts textuels de plus ? Comparer à `margin_100` |

Chaque sélection respecte les cinq catégories du projet. Les panels de 20 sont inclus dans leurs panels de 25. Les panels aléatoires sont reproductibles (graine 42 par défaut). `fit --seed 43 --plan runs/plan43.json` permet d'étudier un autre tirage témoin. Choisir les graines avant de consulter les résultats. Pour exporter les questions d'un protocole en JSONL avec le nombre de répétitions dans le nom du fichier :

```sh
python experiment.py export --arm margin_flips_100 --out runs/prompts
```

## Objets mathématiques

**1. Réponse et ressemblance.** Pour un endpoint $e$, un prompt $p$ et une répétition $r$, $Y_{epr}$ est le texte généré, réalisation de la loi conditionnelle $P_{e,p}$. Une réponse entière constitue une observation. On suppose que les répétitions fournissent des tirages indépendants d’une loi stable pour ce couple. Un cache de réponses ou une modification du service pourrait limiter cette interprétation. Après le nettoyage de méthode 1, on tronque ou complète les chaînes par un symbole spécial jusqu’à une longueur commune $L$. Le noyau de Hamming est

$$ k_L(y,z)=\frac1L\sum_{\ell=1}^{L}\mathbf1[y_\ell=z_\ell]. $$

Deux symboles de remplissage concordent également. Pendant la sélection $L=256$ **caractères**. Comme en méthode 1, pendant le calcul de la distance textuelle finale $L$ est le 95ᵉ centile entier des longueurs dans le lot considéré, borné entre 1 et 1 200.

**2. Écart entre deux lois pour un prompt.** Si $Y,Y'$ sont deux tirages indépendants de $P_{e,p}$ et $Z,Z'$ de $P_{f,p}$, tous indépendants, on définit

$$ d_p(e,f)=\mathrm{MMD}^2_k(P_{e,p},P_{f,p})
 =\mathbb E k(Y,Y')+\mathbb E k(Z,Z')-2\mathbb E k(Y,Z). $$

Dans $\widehat d_p$, chaque espérance intra-endpoint est estimée par la moyenne sur les paires ordonnées de répétitions distinctes (dénominateur $n(n-1)$). Le terme croisé utilise toutes les $nm$ paires. L’estimation peut être négative. Une petite valeur indique une proximité pour les caractéristiques mesurées par ce noyau.

**3. Marge supervisée.** On regroupe les références d’ajustement selon leur vraie identité $(\mathrm{famille},\mathrm{modèle})$. Soit $G_0$ le groupe avec plusieurs déploiements du même modèle. Sur les neuf références, il contient 01, 03, 05, 06 et 08. On estime la nuisance par

$$ N_p=Q_{0.90}\{\widehat d_p(e,f): e,f\in G_0,\ e<f\}. $$

$Q_{0.90}$ est le quantile empirique avec interpolation linéaire. C'est une statistique descriptive de dix paires dépendantes en configuration complète. Pour chaque contraste entre deux modèles distincts $g,h$, on calcule

$$ M_p(g,h)=\operatorname{médiane}_{e\in g,\ f\in h}\widehat d_p(e,f)-N_p. $$

Une marge positive indique que l’écart médian du contraste dépasse le niveau de nuisance estimé. On inclut **tous** les contrastes, y compris LoRA faible contre LoRA fort. Avec les neuf références, cinq modèles donnent dix contrastes. On parcourt les catégories et les rangs en alternant les contrastes dans un ordre fixé ; pour chaque place, on prend le prompt restant de la catégorie ayant la plus grande marge sur le contraste attribué. Les ex æquo suivent l’ordre des identifiants. Une marge négative peut être retenue si tous les candidats restants sont moins bons. C’est un classement exploratoire soumis au risque de sélectionner une fluctuation favorable.

**4. Distance textuelle utilisée pour le clustering.** Le calcul est celui de `méthode 1/stats.py` et `cluster.py`. Dans chaque catégorie $c$, le noyau sur deux observations est $\mathbf1[p=q]k_L(y,z)$. On applique la formule empirique de MMD² à toutes les observations de cette catégorie, obtenant $D_c(e,f)$. Pour estimer le bruit interne, on partage aléatoirement les répétitions de chaque endpoint en deux moitiés à l’intérieur de chaque prompt, 20 fois. Les moyennes et écarts-types ainsi mesurés par endpoint sont agrégés par la médiane, puis multipliés par 0,5 selon l’approximation de méthode 1. Soient $\mu_c,\sigma_c$ ces estimations, avec plancher $10^{-12}$ pour $\sigma_c$. Le score textuel est

$$ T(e,f)=\frac1{|C|}\sum_{c\in C}\frac{D_c(e,f)-\mu_c}{\sigma_c}. $$

$C$ est l’ensemble des catégories présentes. Ces scores standardisés servent au classement. Ils peuvent être négatifs et ne constituent pas des statistiques normales calibrées. $N_p$ mesure la variabilité **entre déploiements** et intervient dans la sélection ; $\mu_c,\sigma_c$ mesurent le bruit **intra-endpoint** et interviennent dans le calcul final.

**5. Empreinte FLIPS adaptée.** Quatre prompts demandent 120 symboles avec les alphabets `A/B`, `0/1`, `X/O` et `+/-`. Après retrait des espaces et retours à la ligne, on exige au moins 100 symboles et un texte constitué uniquement des deux symboles autorisés. On analyse les 100 premiers. Chaque réponse produit un vecteur $X\in\mathbb R^{11}$ contenant la fréquence du deuxième symbole, la fraction d’alternances sur les 99 transitions, la longueur de la plus longue série divisée par 100, et les huit fréquences des triplets sur les 98 fenêtres. Une sortie mal formée consomme sa place et ne fait l’objet d’aucun nouveau tirage.

Pour chaque alphabet disposant d’au moins deux réponses valides de part et d’autre, on estime $\|\mathbb E X_e-\mathbb E X_f\|^2$ par

$$ B_a(e,f)=\|\bar X_e-\bar X_f\|^2-\frac{\mathrm{tr}(S_e)}{n_e}-\frac{\mathrm{tr}(S_f)}{n_f}. $$

$\bar X$, $S$ et $n$ sont la moyenne empirique, la covariance empirique corrigée (diviseur $n-1$) et le nombre de réponses valides pour cet alphabet. La correction soustrait la variance d’échantillonnage des moyennes sous indépendance. Elle porte sur les **réponses**, pas sur les symboles successifs. $B(e,f)$ est la moyenne des $B_a$ disponibles. L’interprétation est conditionnelle aux sorties bien formées. Le rapport conserve le nombre d’alphabets exploitables pour chaque paire.

**6. Fusion et partition.** Sur les paires de calibration, chaque signal est centré par sa médiane et divisé par son écart interquartile $Q_{0.75}-Q_{0.25}$. Pour le signal binaire, seules les paires avec au moins deux alphabets valides entrent dans la calibration. Une fois les échelles gelées, la distance hybride est la moyenne des deux scores standardisés. Si une paire a moins de deux alphabets exploitables, on utilise son score textuel standardisé. Si une échelle de calibration est indéfinie ou quasi nulle, toute la branche hybride utilise le score textuel brut ; `hybrid_active` l’indique. Les indices par paire restent descriptifs et les textes et les suites n’ont pas à être indépendants entre eux pour effectuer cette fusion.

La classification hiérarchique à liaison moyenne fusionne les deux groupes dont la moyenne des distances entre membres est la plus faible. Pour chaque protocole, on choisit sur le **lot de calibration** la coupe qui maximise le MCC : on essaie une partition par intervalle entre hauteurs de fusion distinctes, plus les deux partitions extrêmes. Parmi les seuils maximisants, on retient le candidat médian inférieur. La coupe est ensuite gelée pour le test. Le clustering garantit des groupes transitifs.

**7. Score.** Pour une paire, la classe positive est « même modèle exact ». $TP,TN,FP,FN$ désignent les vrais positifs, vrais négatifs, fausses fusions et fausses séparations. Le score local reproduit la convention du concours :

$$ \mathrm{MCC}_{+}=\max\left(0,\frac{TP\,TN-FP\,FN}
{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}\right). $$

Un dénominateur nul donne 0. `test` produit 36 décisions sur paires non ordonnées (10 positives, 26 négatives), les CSV des partitions et un JSON détaillant les erreurs de chaque protocole. `loo` retire successivement un endpoint pendant **toute** la sélection et la calibration, refait le clustering sur les neuf à partir du lot de test et évalue uniquement les huit paires contenant l’endpoint caché. Le rapport agrégé contient 72 décisions orientées de validation croisée ; chaque paire intervient deux fois sous deux ajustements distincts. Elles sont dépendantes. Ce MCC agrégé et le MCC à 36 paires répondent à deux questions différentes. Une seule famille étant observée, aucun MCC inter-familles n’est estimé.

## Fichiers

| Fichier | Rôle |
|---|---|
| `design.py` | Catalogue des prompts, identifiants de référence, bornes entre lots, contrôle des doublons et de la complétude. |
| `collect.py` | Collecte reprenable, avec le client du projet ; seules les neuf références figurent dans la liste des cibles. Verrou local pour éviter deux collectes simultanées dans le même fichier. |
| `signals.py` | Marges MMD par contraste, tirages témoins, sélection des prompts et calcul des caractéristiques FLIPS. |
| `experiment.py` | Commandes `fit`, `test`, `loo`, `export`, calibration des seuils, clustering, MCC et rapports locaux. |
| `test_experiment.py` | Tests hors réseau et générateur artificiel reproductible, pour vérifier les formules et la séparation des lots. |
| `.gitignore` | Exclut du dépôt les générations et les sorties de calcul dans `runs/`. |

## Vérification hors ligne et résultats simulés

```sh
python -m unittest -v test_experiment
python test_experiment.py --demo runs/demo/answers_simulees.jsonl
python experiment.py fit --answers runs/demo/answers_simulees.jsonl --plan runs/demo/plan.json
python experiment.py test --answers runs/demo/answers_simulees.jsonl --plan runs/demo/plan.json --out runs/demo/resultats
python experiment.py loo --answers runs/demo/answers_simulees.jsonl --out runs/demo/resultats
```

Le générateur construit artificiellement un effet de modèle dépendant du prompt, un préambule propre à un déploiement et des biais binaires dépendant du modèle. Les noms `ref-*` dans la simulation identifient des rôles fictifs. Résultats du test technique avec la graine par défaut :

| Protocole | MCC sur réponses simulées réservées | MCC lorsque l’endpoint est caché au réglage |
|---|---:|---:|
| Méthode 1 (100) | 0,721 | 0,721 |
| Aléatoire (100) | 0,721 | 0,721 |
| Marge (100) | 1,000 | 0,372 |
| Aléatoire (80) | 0,721 | 0,721 |
| Marge (80) | 1,000 | 0,372 |
| Marge + FLIPS (100) | 1,000 | 0,567 |

Ces nombres vérifient le parcours logiciel et illustrent le risque de surajustement à des endpoints connus. La performance sur le service de référence nécessite ses générations réelles.

**Sources.** [Gao, Liang et Guestrin — Model Equality Testing](https://arxiv.org/abs/2410.20247) · [Richardeau et al. — FLIPS](https://proceedings.mlr.press/v306/gurvan26a.html). La sélection par marge et la fusion décrites ici sont des adaptations au défi.
