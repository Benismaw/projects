# Décodeur JPEG


## Résumé et description

Ce projet implémente un décodeur JPEG en C qui transforme des images JPEG (.jpg) en format PPM/PGM.  
Notre décodeur prend en charge :

- Images en niveaux de gris (8×8 et multi-blocs) avec sortie `.pgm`
- Images en couleur avec sortie `.ppm`
- Sous-échantillonnage chroma (formats 4:2:0, 4:2:2, etc.)
- DCT rapide avec l'algorithme de Loeffler

## Auteurs

**Équipe**: Wajd, Nasma et Sam
## Répartition des tâches

| Membre | Rôles principaux             | Tâches                                                              |
|--------|------------------------------|---------------------------------------------------------------------|
| Nasma   | Infrastructure de décodage   | Gestion des tables de Huffman, décodage multi-MCU, écriture du PPM, optimisation iDCT |
| Wajd  | Traitement d'image    | Implémentation iDCT, conversion YCbCr→RGB , parcours Zig-zag & quantification inverse, extraction des blocs, upsampling                          |
| Sam    | Parsing et intégration         | Lecture des entêtes JPEG, bitstream parsing, intégration des modules, tests          |

## Rôles communs partagés par tous :
- Conception et coordination globale
- Généralisation du décodeur

## Organisation de travail

Dès les premières étapes du projet, nous avons réparti les tâches de manière structurée, en commençant par invaders. Progressivement, avec la généralisation du décodeur pour intégrer l’échelle de gris, la troncature, les couleurs et le sous-échantillonnage, notre organisation s’est naturellement orientée vers une dynamique plus collective : chacun intervenait là où il y avait besoin, et les contributions s’imbriquaient avec fluidité.
Nos échanges constants via Discord, complétés par des rencontres presque quotidiennes y compris les weekends ont joué un rôle clé dans cette dynamique favorisant une progression continue et coordonnée.

## Compétences techniques acquises

- Compréhension approfondie du format JPEG et de ses composants
- Manipulation de bits et d'octets au niveau système
- Algorithmes de compression/décompression
- Traitement du signal (DCT/iDCT)
- Colorimétrie et conversion entre espaces colorimétriques
- Gestion de projet et pratiques de développement collaboratif
- Programmation en C : allocation mémoire, gestion des pointeurs, structures de données
- Modélisation visuelle de processus complexes à l'aide de schémas explicatifs, facilitant l'analyse et la mise en œuvre d'algorithmes

## Architecture du code

```
                ┌────────────────┐
                │  Fichier JPEG  │
                └────────┬───────┘
                         ▼
┌────────────────────────────────────────────────┐
│  lecture_entete.c     ┌───────────────────────────────────┐
│  ┌─────────────────┐  │ ┌───────────────────┐  │          │
│  │ Quantification  │──┘ │ Tables de Huffman │────┐        │
│  │    Tables       │    │      (DHT)        │  │ │        │
│  └─────────────────┘    └───────────────────┘  │ │        │
│                                                │ │        │
│  ┌───────────────────────────────┐             │ │        │
│  │  Flux de données compressées  │             │ │        │
│  └───────┬───────────────────────┘             │ │        │
└──────────┼─────────────────────────────────────┘ │        │
           ▼                                       ▼        │
┌────────────────────────┐       ┌──────────────────────┐   │
│     bitstream.c        │       │      arbre.c         │   │
│ Lecture bit par bit du │       │ Création des arbres  │   │
│   flux compressé       │       │      de Huffman      │   │
└──────────┬─────────────┘       └──────────┬───────────┘   │
           ▼                                ▼               │
┌────────────────────────────────────────────────┐          │
│                  extract.c                     │          │
│   Décodage Huffman et extraction coefficients  │          │
└──────────────────────┬─────────────────────────┘          │
                       ▼                                    │
┌────────────────────────────────────────────────┐          │
│                  zigzag.c                      │          │
│         Réarrangement inverse zigzag           │          │
└──────────────────────┬─────────────────────────┘          │
                       ▼                                    │
┌────────────────────────────────────────────────┐          │
│                  quantif.c                    ◄───────────┘
│           Quantification inverse               │
└──────────────────────┬─────────────────────────┘
                       ▼
┌────────────────────────────────────────────────┐
│               fast_idct.c / idct.c             │
│        Transformation cosinus inverse          │
└──────────────────────┬─────────────────────────┘
                       ▼
┌────────────────────────────────────────────────┐
│                 upsample.c                     │
│       Up-sampling (si sous-échantillonné)      │
└──────────────────────┬─────────────────────────┘
                       ▼
┌────────────────────────────────────────────────┐
│                conversion.c                    │
│         Conversion YCbCr vers RGB              │
└──────────────────────┬─────────────────────────┘
                       ▼
┌────────────────────────────────────────────────┐
│                 ecriture.c                     │
│    Écriture du fichier PPM/PGM de sortie       │
└──────────────────────┬─────────────────────────┘
                       ▼
                ┌───────────────┐
                │Fichier PPM/PGM│
                └───────────────┘
```

### Description des modules

#### `lecture_entete.c`

- **Entrée**: Fichier JPEG  
- **Sortie**: Structure `parsed_file` contenant toutes les métadonnées  
- Parse les marqueurs JPEG (SOI, APP0, COM, DQT, SOF, DHT, SOS, EOI)
- Extrait les tables de quantification
- Récupère les informations d'image (dimensions, composantes)
- Construit les arbres de Huffman
- Stocke les données compressées

#### `bitstream.c`

- **Entrée**: Données compressées de l'image  
- **Sortie**: Bits individuels ou séquences de bits  
- Gère la lecture bit à bit du flux de données
- Gère le "byte stuffing" (séquences FF00)

#### `arbre.c`

- **Entrée**: Tables de Huffman (DHT)  
- **Sortie**: Arbres de Huffman pour décodage  
- Construction des arbres à partir des tables
- Navigation dans les arbres pour le décodage

#### `extract.c`

- **Entrée**: Flux de bits, arbres de Huffman  
- **Sortie**: Vecteurs de coefficients DCT  
- Décode les symboles Huffman
- Reconstruit les coefficients DC et AC
- Gère le RLE (Run-Length Encoding)

#### `zigzag.c`

- **Entrée**: Vecteurs de coefficients  
- **Sortie**: Matrices 8×8 de coefficients  
- Réarrange les coefficients de l'ordre zigzag à l'ordre matriciel

#### `quantif.c`

- **Entrée**: Vecteurs de coefficients, tables de quantification  
- **Sortie**: Coefficients déquantifiés  
- Multiplie chaque coefficient par sa valeur de quantification

#### `idct.c` / `fast_idct.c`

- **Entrée**: Matrices de coefficients DCT  
- **Sortie**: Blocs de pixels  
- Implémente la Transformée en Cosinus Discrète Inverse
- Utilise l'algorithme de Loeffler pour une meilleure performance

#### `upsample.c`

- **Entrée**: MCUs avec composantes sous-échantillonnées  
- **Sortie**: MCUs avec composantes à pleine résolution  
- Effectue l'upsampling horizontal et/ou vertical selon le format

#### `conversion.c`

- **Entrée**: Pixels en format YCbCr  
- **Sortie**: Pixels en format RGB  
- Convertit l'espace colorimétrique YCbCr en RGB

#### `ecriture.c`

- **Entrée**: Blocs de pixels RGB ou niveaux de gris  
- **Sortie**: Fichier PPM/PGM  
- Écrit l'entête PPM/PGM
- Écrit les données pixel par pixel
- Gère les cas de troncature d'image

## Compilation et utilisation

### Compilation

```bash
make
```

### Utilisation

```bash
./jpeg2ppm <chemin_vers_image_jpeg>
```

#### Options disponibles

| Option | Description |
|--------|-------------|
| `-v` | Mode verbeux - Affiche des informations détaillées sur le parsing |
| `-h` | Affiche les arbres de Huffman utilisés pour le décodage |
| `--outfile=<fichier>` | Spécifie un nom personnalisé pour le fichier de sortie |

#### Formats de sortie

Le décodeur génère automatiquement :
- Des fichiers `.pgm` pour les images en niveaux de gris
- Des fichiers `.ppm` pour les images en couleur

## Tests effectués

| Image                | Type             | Caractéristiques                  |
|----------------------|------------------|-----------------------------------|
| `app1.jpeg`         | Niveaux de gris | Segment APP1 non supporté, gestion d'erreur et libération mémoire propre |
| `art_nasma.jpg`          | Couleur sans sous-échantillonnage | Simple bloc 8×8                  |
| `blanc.png`             | Fichier de type incorrect  | Test de gestion d'erreur et libération mémoire propre |
| `bw_degrade.jpg`          | Format JPEG progressif  | Test de détection de format non supporté et libération mémoire propre |
| `cat_tronc_droite.jpg`   | Niveaux de gris  | Troncature à droite                  |
| `country.jpeg`            | Couleur avec sous-échantillonnage vertical         | 1 seule MCU de taille 2x2 |
| `eagle.jpg`      | Couleur avec sous-échantillonnage horizontal | Troncature à droite de la Mcu, mais pas d'un bloc (le bloc est aligné) |
| `empty.jpg` | Niveaux de gris | Test avec fichier sans données d'entropie (pas de flux compressé) |
| `gris_64_64_debug.jpg` & `gris_320_320_debug.jpg` | Niveaux de gris | Reproduire et isoler les problèmes rencontrés avec `gris.jpg`|
| `horizontal_no_tronc.jpg` | Couleur avec sous-échantillonnage horizontal | `horizontal.jpg` mais avec dimensions multiples de 8 pour déboguer |
| `koulthoum_tronc_bas.jpg` | Niveaux de gris | Test de troncature en bas de l'image |
| `milano.jpg` | Couleur avec sous-échantillonnage horizontal | Test avec mêmes dimensions et sous-échantillonnage que `horizontal.jpg` |
| `noir.jpg` | Couleur avec sous-échantillonnage horizontal et vertical | Test avec troncature à droite |
| `perle.jpg` | Couleur avec sous-échantillonnage horizontal | Image de petite taille (64×72 pixels) |
| `pixilaa.jpg` | Niveaux de gris | Image minimale de taille 1×1 pixel |
| `squirrel.jpg` | Couleur avec sous-échantillonnage horizontal | Image de petite taille carrée (64×64 pixels) |
| `zebra.jpeg` | Couleur avec sous-échantillonnage horizontal et vertical | Motif de rayures pour tester le calcul des coefficients DC |

### Scripts de test

Deux scripts shell facilitent nos tests automatisés :

- **`run_all_images.sh`** : Compile le projet et traite tous les fichiers du dossier `images/`, vérifiant la robustesse du décodeur.

- **`check_memory_leaks.sh`** : Utilise Valgrind pour détecter les fuites mémoire sur chaque fichier du dossier `images/`.

## Problèmes rencontrés et solutions

### Compréhension des coefficients DC et AC

- **Problème** : Difficulté à comprendre comment les coefficients DCT sont encodés  
- **Solution** : Étude détaillée de la spécification JPEG et implémentation pas à pas

### Corruption de sortie pour images en niveaux de gris

- **Problème** : Images en sortie avec faible contraste, apparaissant grises  
- **Solution** : Correction des facteurs d'échelle dans l'iDCT et normalisation appropriée

### Upsampling des composantes chromatiques

- **Problème** : Difficultés avec les différents formats de sous-échantillonnage  
- **Solution** : Approche générale qui traite individuellement chaque cas (4:2:0, 4:2:2, etc.)

### Écriture des MCUs

- **Problème** : Troncature incorrecte des images aux bords  
- **Solution** : Implémentation de fonctions dédiées pour gérer la troncature des blocs

## Améliorations futures

- **Support du JPEG progressif**  
  Actuellement non supporté, cela permettrait d'afficher progressivement l'image

- **Optimisation mémoire**  
  Réduire l'empreinte mémoire pour les grandes images

- **Stéganographie**  
  Possibilité de cacher/récupérer des informations dans les coefficients DCT

- **Interface graphique**  
  Ajouter une interface pour visualiser les différentes étapes du décodage

- **Formats de sortie supplémentaires**  
  Ajouter le support de formats comme PNG, BMP, etc.

## Mot du groupe

Ce projet a été une belle occasion de renforcer notre cohésion tout en approfondissant notre compréhension des algorithmes de compression d’images, notamment du format JPEG.
Face à sa complexité, nous avons su conjuguer nos efforts, partager nos savoirs et avancer ensemble de manière rigoureuse.
Nous sommes fiers du travail accompli, reflet de notre collaboration efficace et des compétences techniques développées tout au long du projet.

