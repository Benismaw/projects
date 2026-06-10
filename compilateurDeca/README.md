# Compilateur Deca

Compilateur complet pour le langage **Deca** (sous-ensemble de Java), développé en équipe de 5 personnes dans le cadre du projet emblématique de l'ENSIMAG.

> Objectif : produire un logiciel "zéro défaut" capable de traduire du code source complexe en assembleur pour une machine abstraite (IMA).

## Pipeline de compilation

Le compilateur couvre les trois étapes classiques :

### 1. Analyse (ANTLR4)
Transformation du code source en **Arbre de Syntaxe Abstraite (AST)** via ANTLR4.

### 2. Vérifications sémantiques
3 passes de vérification contextuelle pour décorer l'AST :
- Hiérarchie des classes
- Signatures des membres
- Corps des méthodes

### 3. Génération de code
Utilisation du patron de conception **Interprète** pour traduire chaque nœud de l'AST en instructions IMA, avec gestion manuelle des :
- **VTables** (liaison dynamique)
- **Blocs d'activation** dans la pile

## Défis techniques

**Gestion des ressources :**
- `RegManager` — gestion des registres avec mécanisme de **spilling** (sauvegarde sur pile quand les registres sont saturés)
- `StackManager` — calcul dynamique du TSTO pour prévenir les débordements à l'exécution

**Compilation parallèle :**
Support de l'option `-P` (compilation parallèle) en bannissant toute variable `static` — chaque instance du compilateur est totalement isolée.

## Méthodologie & Qualité

**TDD strict** — aucun code intégré sans suite de tests validée.

Stratégie de tests sur 3 axes :
- Hiérarchie / VTable
- Allocation dans le tas
- Blocs d'activation

**Résultats de couverture (JaCoCo) :**

| Paquetage | Couverture |
|-----------|------------|
| Sémantique | **86%** |
| AST | **82%** |

## Tech Stack

Java, ANTLR4, Maven, JaCoCo, Git

## Ce que ce projet m'a appris

- Gestion d'un cahier des charges strict sous contrainte de temps
- Travail en équipe Agile avec Git et Maven
- Conception logicielle rigoureuse (TDD, patrons de conception)
- Réflexion sur l'optimisation du code assembleur produit
