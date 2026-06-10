#ifndef ARBRE_H
#define ARBRE_H

#include <stdint.h>
#include <stdbool.h>

typedef struct HuffmanTable{
    uint8_t dc_ac;          // 0 (DC) ou 1 (AC)
    uint8_t index;          // Indice de la table (entre 0 et 3)
    uint8_t offsets[16];    // Nombre de codes pour chaque longueur (de 1 à 16 bits)
    uint16_t sum_offsets;   // Nombre total de codes (somme des éléments de offsets)
    uint8_t *valeurs;       // Symboles associés aux codes
} HuffmanTable;

typedef struct Node {
    uint8_t cle;
    struct Node* fg;
    struct Node* fd;
} Node;

typedef struct huffman_trees{  
    Node *dc[4];
    Node *ac[4];
} huffman_trees;

// Fonction pour construire un arbre de Huffman à partir d’une table Huffman
Node* decode_huff(HuffmanTable *table);

// Libère l’ensemble des arbres AC/DC pour les 4 tables possibles
void free_huffman_trees(huffman_trees *tree);

// Libère récursivement la mémoire occupée par l’arbre de Huffman
void free_huffman_tree(Node *node);

// Affiche les chemins de l’arbre de Huffman en parcours en largeur (BFS)
void print_huffman_paths_bfs(Node *root);

#endif // ARBRE_H