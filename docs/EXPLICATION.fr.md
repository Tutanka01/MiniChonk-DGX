# Comment Mistral Large 4 tourne sur deux GB10

Ce document explique tout le projet : le problème, les concepts, ce qui a été essayé, ce qui a échoué et pourquoi la
configuration finale est celle-là. Sauf mention contraire, les chiffres sont mesurés sur le cluster, et les sorties
brutes des benchs de qualité sont dans [`results/`](../results).

## En bref

| Question | Réponse |
|---|---|
| Le modèle complet tient-il ? | Non. Le plus petit GGUF pèse 233 Go, il faudrait environ 114,5 Gio par machine |
| Qu'est-ce qui tourne ? | 112 experts sur 128 par couche, environ 1,7 bit par poids routé, 206 Go |
| À quelle vitesse ? | 9 à 11,5 tok/s en génération, 135 tok/s pour lire un long prompt |
| Quelle qualité ? | Perplexité 5,31, MMLU 47 %, ARC 63 %, 13/13 sur des questions faciles. Perte non chiffrée, faute de référence |

## 1. Le problème

Mistral Large 4 a environ 1 000 milliards de paramètres, mais n'en utilise que 52 milliards pour chaque token. Il
garde tout de même l'ensemble en mémoire, car il ne sait pas à l'avance lesquels serviront.

Chaque machine voit 119 Gio, dont environ 115 disponibles. Les deux ensemble offrent donc environ 230 Gio, et la
mémoire n'est pas mise en commun : il faut découper le modèle en deux moitiés.

## 2. Les concepts

### Mixture of Experts (MoE)
Dans un modèle classique, chaque couche applique le même réseau à tous les tokens. Ici, 71 couches sur 72 contiennent
128 petits réseaux, les **experts**. Un **routeur** choisit 4 experts par token, auxquels s'ajoute un expert partagé
toujours actif. D'où l'écart entre 1T et 52B.

Répartition mesurée dans les en-têtes de `UD-IQ1_S` :

| Partie | Paramètres | Taille | Utilisé à chaque token |
|---|---|---|---|
| Experts routés | 1 029 B | 214,0 Go | 4 sur 128, soit 6,7 Go |
| Expert partagé | 8,0 B | 8,5 Go | Tout |
| Attention et normalisation | 9,1 B | 7,7 Go | Tout |
| Embeddings, sortie, reste | 3,0 B | 2,8 Go | En grande partie |

Les experts routés représentent 92 % du poids mais seulement un quart de ce qui est lu à chaque token. Deux
conséquences en découlent :
- Retirer des experts libère beaucoup de mémoire.
- Retirer des experts n'accélère presque rien.

### Quantification
Stocker chaque poids sur moins de bits. Le nom du fichier en donne le niveau. Calculé sur les tailles réelles, les
experts routés sont à environ 1,7 bit par poids dans `UD-IQ1_S` et 2,7 bits dans `UD-Q2_K_XL`. Unsloth garde les parties
critiques en haute précision et n'écrase que les experts routés.

### GGUF et découpe par tranches
Un GGUF contient un en-tête, puis les tenseurs à la suite. Un tenseur d'experts a la forme `[6144, 6144, 128]`, et la
dernière dimension est stockée de façon contiguë : **chaque expert est une tranche d'octets**. On peut donc en retirer
sans décoder ni recalculer la quantification.

Le routeur (`ffn_gate_inp`) a lui aussi une dimension par expert et doit être découpé avec le même masque. llama.cpp
lit ensuite le nombre d'experts dans les métadonnées, qu'il suffit de modifier.

Un piège a été évité : `attn_v_b`, `attn_k_b` et `indexer.*` ont aussi une dimension de 128 sans aucun rapport avec les
experts. L'outil cible donc les tenseurs par nom exact, jamais par forme.

### AIMER
Une méthode pour choisir quels experts retirer, [publiée en 2026](https://arxiv.org/abs/2603.18492). Elle calcule un
score à partir des seuls poids, sans données. Le projet [LeTrim](https://huggingface.co/akhaliq/Mistral-Large-4-1T-A52B-LeTrim-NVFP4)
publie ces scores pour ce modèle : on retire les experts aux scores les plus élevés.

### RPC, RDMA et répartition
- **RPC** : llama.cpp envoie des morceaux du calcul à un `ggml-rpc-server` sur l'autre machine. Le serveur n'ayant pas
  d'authentification, il n'écoute que sur le lien direct.
- **RDMA** : transfert direct entre mémoires, sans passer par la pile TCP.
- **Répartition `layer`** : un nœud calcule ses couches, puis passe la main à l'autre. Les machines travaillent **à
  tour de rôle**.

### DSpark
Un tout petit modèle (2,2 Go) propose 3 tokens d'avance, que le grand modèle valide en une seule passe. C'est rentable
car lire un lot n'est pas beaucoup plus coûteux que lire un token. La sortie du grand modèle reste celle qu'il aurait
produite.

### Les mesures de qualité
- **Perplexité** : à quel point le modèle est surpris par un vrai texte. Plus bas = mieux. Elle est très sensible et
  mesurée ici sur 192 ko d'anglais, de français, de Python et de texte technique.
- **MMLU** : QCM de connaissances. Le hasard donne 25 %, marge d'environ ± 3,5 points sur 200 questions.
- **ARC challenge** : QCM de raisonnement scientifique, mêmes marges.

Ces scores ne se comparent qu'entre eux : ce protocole n'est pas celui des classements publics.

## 3. Le matériel et le réseau

Les machines sont des ThinkStation PGX, avec le même GB10 qu'un DGX Spark. Chaque port physique apparaît sous forme de
plusieurs interfaces, sur deux sous-réseaux.

| Test | Résultat |
|---|---|
| TCP, un flux | 14 à 27 Gb/s |
| TCP, 4 flux | 80 à 96 Gb/s |
| RDMA, une interface | 109 Gb/s sur 3 interfaces, pas de résultat sur `rocep1s0f0` |
| RDMA, trois interfaces simultanées | 159 Gb/s au total |

La paire retenue est la plus rapide mesurée, et le routage du noyau y passe par la bonne interface. `cluster.sh check`
contrôle ce point.

Bench de la répartition sur un Qwen de 19 Go :

| Configuration | Génération | Lecture du prompt |
|---|---|---|
| Un seul nœud | 11,06 tok/s | 932 tok/s |
| Deux nœuds, `layer`, RDMA | 10,43 tok/s | 909 tok/s |
| Deux nœuds, `layer`, TCP | 10,44 tok/s | 867 tok/s |
| Deux nœuds, `tensor` | 6,31 tok/s | 529 tok/s |
| Deux nœuds, `row` | Refusé en RPC | – |

**Le second nœud sert à faire tenir le modèle, pas à aller plus vite**, et le réseau n'est pas le facteur limitant de
la génération. C'est la bande passante mémoire qui limite.

## 4. Comment la configuration finale a été trouvée

### Vérifier l'ordre des experts
Le masque LeTrim a été calculé sur un autre fichier que notre GGUF. Si l'expert n°7 n'était pas le même des deux côtés,
le modèle chargerait mais serait dégradé sans aucun message d'erreur.

`verify_expert_order.py` compare les bits de signe des poids entre les fichiers NVFP4, FP8 et GGUF, sur les couches 1,
36 et 71. Résultat : 100 % de concordance pour un même expert, environ 50 % entre experts différents, et le bon
expert retrouvé pour 128 experts sur 128. Le test ne télécharge que quelques Mo.

### Tester une idée, et se tromper
Le fichier imatrix publié contient le nombre de passages par chaque expert, sur 295 000 tokens. L'idée était de garder
les experts les plus utilisés : à 64 experts, ils conservent 73,5 % du routage, contre 56,4 % pour AIMER.

**Cette idée s'est révélée moins bonne :** MMLU 38,0 % contre 45,5 %, ARC 43,7 % contre 54,9 %. L'explication la plus
probable, non démontrée, est que les experts peu routés portent des connaissances spécialisées.

### Experts ou bits ?

| Passage | Perplexité | MMLU | ARC (142 questions) |
|---|---|---|---|
| 64 experts, 1,7 → 2,7 bits | 11,12 → 8,63 | 45,5 → 50,0 % | 54,9 → 55,6 % |
| 1,7 bit, 64 → 96 experts | 11,12 → 6,45 | 45,5 → 48,0 % | 54,9 → 57,0 % |
| 1,7 bit, 96 → 112 experts | 6,45 → 5,31 | 48,0 → 47,0 % | 57,0 → 60,0 % |

À mémoire comparable, 112 experts à 1,7 bit (206 Go) bat 64 experts à 2,7 bits (200 Go) sur la perplexité et sur ARC.
Sur MMLU, c'est l'inverse (47,0 % contre 50,0 %), mais l'écart reste dans la marge d'erreur.

Le gain est visible aussi à l'usage : à 64 experts, le modèle écrivait `36 - 7 = 35` et déformait des mots français.
À 96 experts, ces erreurs ont disparu des tests effectués.

Perplexité par domaine :

| Variante | Python | Technique | Français | Littérature anglaise |
|---|---|---|---|---|
| 64 experts AIMER | 3,45 | 4,81 | 25,94 | 29,45 |
| 96 experts AIMER | 2,73 | 3,78 | 10,10 | 16,07 |
| 112 experts AIMER | 2,31 | 2,68 | 8,79 | 13,64 |

### La vitesse
Avec DSpark, à 96 experts : de 6,33 à 10,46 tok/s, soit +65 %, avec un taux d'acceptation de 55 à 73 %. Le gain varie
selon le texte : le code et le texte prévisible vont plus vite. À 112 experts, on mesure 9,2 à 11,5 tok/s selon les
prompts. Passer de 96 à 112 experts ne coûte que peu de vitesse.

## 5. L'incident mémoire et ce qu'il a changé

Un test du modèle complet a saturé la mémoire du second nœud pendant 5 minutes. Le noyau a tué 28 petits processus de
session, y compris d'un autre compte. Trois erreurs se sont cumulées :

1. **Lancer un chargement à la limite** : 114,5 Gio demandés pour environ 115 disponibles.
2. **Un garde-fou inadapté** : la mémoire GPU est verrouillée et n'apparaît pas dans le RSS d'un processus. Le noyau
   tue donc de petits processus innocents, et une surveillance toutes les secondes arrive trop tard.
3. **Une boucle sans arrêt** : l'évaluation a relancé le chargement après l'échec.

Protections actuelles :
- `cluster.sh preflight` refuse tout modèle qui laisserait moins de 6 Gio, d'après la mémoire réellement disponible.
- Les processus d'inférence ont `oom_score_adj=1000` : le noyau les tue en premier s'il doit choisir. Ce réglage n'a
  pas été testé sur une vraie saturation.
- L'évaluation s'arrête au premier échec.

Autre leçon : les estimations de mémoire étaient fausses. Le banc QCM consommait environ 20 Gio de plus que prévu avec
de gros lots. Les valeurs de `cluster.sh` sont donc mesurées, et chaque montée en taille est d'abord mesurée sur une
variante plus petite.

## 6. Le déroulé technique

```
Hugging Face                 Nœud 1 (head)                         Nœud 2 (worker)
UD-IQ1_S, 233 Go  ──fetch──► source vérifiée en sha256
masque AIMER      ──trim───► IQ1_S-aimer112, 206 Go
                             vérifié octet par octet
                             llama-server :8081  ◄──RDMA 200G──►  ggml-rpc-server
                             + brouillon DSpark                   sans poids sur disque
                             ~50 % des couches                    ~50 % des couches
```

| Fichier | Rôle |
|---|---|
| `minichonk/gguf.py` | Lit l'en-tête et prouve que les tailles calculées tombent exactement sur les offsets du fichier |
| `minichonk/trim.py` | Découpe, refuse une source incomplète, un masque différent ou un disque trop petit, et reprend après interruption |
| `scripts/cluster.sh` | Point d'entrée unique |
| `scripts/fetch_model.py` | Téléchargement avec reprise et contrôle sha256 |
| `scripts/run_eval.sh` | Bench de qualité |
| `scripts/verify_expert_order.py` | Vérification de l'ordre des experts |
| `scripts/expert_usage.py`, `make_*_mask.py` | Génération des masques |
| `tests/` | 33 tests : découpe, corruption, reprise, masques, scores publiés, arrêt des processus |

Pour rejouer l'ensemble, suivre la section « From zero » du [README](../README.md).

## 7. Les limites

- **Pas de référence** : le modèle complet ne tient pas, donc la perte n'est pas chiffrée. Un MMLU à 47 % reste faible.
- **Conditions de mesure inégales** : les paramètres du banc QCM ont changé en cours de route pour réduire la mémoire
  (voir `results/runs.json`). L'équivalence n'a été vérifiée que sur les 20 premières questions de MMLU. Deux variantes
  ont un ARC sur 142 questions seulement.
- **Pas d'évaluation sur des tâches réelles** : 13/13 porte sur des questions faciles.
- **Une seule requête à la fois**, et aucune mesure sur plusieurs heures.
- **Mémoire serrée** : 7 Gio disponibles sur le worker et 15 sur le head. Charger un autre modèle sur ces machines pendant
  que ce serveur tourne peut saturer la mémoire.

## 8. Pistes non testées

- **Réduire la précision de l'expert partagé et de l'attention.** Ils représentent 16 Go lus à chaque token. Les passer
  à environ 4,5 bits pourrait donner autour de +30 % de vitesse (estimation) et libérer de quoi garder quelques
  experts de plus. Le risque pour la qualité est réel, car ces parties servent à chaque token.
- **Mélanger les quantifications tenseur par tenseur**, en donnant plus de bits aux experts les plus importants.
- **Combiner AIMER et usage réel** dans un même score.

## 9. Faut-il publier les poids sur Hugging Face ?

**Non, ce n'est pas nécessaire, et je ne le recommande pas pour l'instant.**

Le dépôt suffit pour reproduire le modèle : le source GGUF est public, et le masque est inclus. La différence pour un
utilisateur est faible :

| | Sans poids publiés | Avec poids publiés |
|---|---|---|
| Téléchargement | 235 Go | 208 Go |
| Étape de découpe | 10 minutes, une commande | Aucune |
| Lancement sur deux nœuds | Identique | Identique |

Publier n'économise donc que 27 Go et 10 minutes, et tout le reste de la configuration reste nécessaire.

**Ce que la licence permet :** redistribuer et modifier sont autorisés, à condition d'inclure la notice de Mistral. Un
service commercial de type *Model as a Service* exige une licence commerciale distincte, ce qui concerne aussi les
utilisateurs.

**Pourquoi attendre :** une fois publié avec le nom Mistral Large 4, un modèle à 1,7 bit sera jugé sur sa réputation,
alors que sa qualité n'a été mesurée que sur des benchs comparatifs et des questions faciles.

**Quand ce serait utile :** si le modèle est validé sur de vrais usages et si la visibilité est l'objectif, publier
uniquement `aimer112`, avec la notice, le lien vers ce dépôt et une fiche qui annonce clairement la perte de qualité.
