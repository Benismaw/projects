  
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "arbre.h"

// Fonction pour créer un nouveau nœud avec une valeur donnée
Node* creer_node(uint8_t valeur) {
    Node* new_node = (Node*)malloc(sizeof(Node));
    if (!new_node) {
        fprintf(stderr, "Node Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    new_node->cle = valeur;
    new_node->fg = NULL;
    new_node->fd = NULL;
    return new_node;
}

// Fonction pour construire un arbre de Huffman à partir d’une table Huffman
Node* decode_huff(HuffmanTable *table) {
    if (table == NULL){
        return NULL;  // Retourne NULL si la table est invalide
    }
    Node* arbre_huff = creer_node(0); // Crée la racine de l’arbre
    uint64_t code = 0;
    uint8_t val_indice = 0;
    
    // Parcourt chaque longueur de code (de 1 à 16 bits)
    for (uint8_t i = 0; i < 16; i++) {
        uint8_t nb_valeurs = table->offsets[i]; // Nombre de symboles ayant un code de longueur i+1
        uint64_t code_longueur = i+1;
        for (uint16_t j = 0; j < nb_valeurs; j++) {
            uint8_t valeur = table->valeurs[val_indice++]; // Récupère la valeur associée
            Node* courant = arbre_huff;

            // Crée le chemin correspondant au code binaire
            for (uint8_t nbr_bit = code_longueur; nbr_bit > 0; nbr_bit--) {
                if ((code >> (nbr_bit - 1)) & 1) { // Si le bit est 1 (vers la droite)
                    if (!courant->fd)
                        courant->fd = creer_node(-1);
                    courant = courant->fd;
                } else { // Sinon (bit 0, vers la gauche)
                    if (!courant->fg)
                        courant->fg = creer_node(-1);
                    courant = courant->fg;
                }
            }
            courant->cle = valeur; // Associe la valeur au nœud feuille
            code++; // Incrémente le code
        }
        code <<= 1;// Décale à gauche pour préparer la prochaine longueur
    }
    return arbre_huff;
}

// Affiche les chemins de l’arbre de Huffman en parcours en largeur (BFS)
void print_huffman_paths_bfs(Node *root) {
    if (!root) return;

    // Définition de la structure de file pour BFS
    typedef struct {
        Node *node;
        char path[32];
        int depth;
    } QueueElem;

    QueueElem queue[1024];  // File de taille fixe
    int front = 0, rear = 0;

    // Ajoute la racine à la file
    queue[rear++] = (QueueElem){root, "", 0};

    while (front < rear) {
        QueueElem current = queue[front++];
        Node *node = current.node;

        // Si c'est une feuille, on affiche le chemin et le symbole
        if (!node->fg && !node->fd) {
            if (current.depth == 0)
                printf("\tpath: - symbol: %x\n", node->cle);
            else
                printf("\tpath: %s symbol: %x\n", current.path, node->cle);
        }

        // Ajoute le fils gauche à la file avec le chemin mis à jour
        if (node->fg) {
            QueueElem left = {node->fg, "", current.depth + 1};
            size_t len = strlen(current.path);
            if (len < sizeof(left.path) - 1) {
                strcpy(left.path, current.path);
                left.path[len] = '0';
                left.path[len + 1] = '\0';
            } else {
                strncpy(left.path, current.path, sizeof(left.path) - 2);
                left.path[sizeof(left.path) - 2] = '0';
                left.path[sizeof(left.path) - 1] = '\0';
            }
            queue[rear++] = left;
        }
        // Ajoute le fils droit à la file avec le chemin mis à jour
        if (node->fd) {
            QueueElem right = {node->fd, "", current.depth + 1};
            size_t len = strlen(current.path);
            if (len < sizeof(right.path) - 1) {
                strcpy(right.path, current.path);
                right.path[len] = '1';
                right.path[len + 1] = '\0';
            } else {
                strncpy(right.path, current.path, sizeof(right.path) - 2);
                right.path[sizeof(right.path) - 2] = '1';
                right.path[sizeof(right.path) - 1] = '\0';
            }
            queue[rear++] = right;
        }
    }
    printf("\n");
}

// Libère récursivement la mémoire occupée par l’arbre de Huffman
void free_huffman_tree(Node *node) {
    if (node != NULL) {
        free_huffman_tree(node->fg);
        free_huffman_tree(node->fd);
        free(node);
    }
}
// Libère l’ensemble des arbres AC/DC pour les 4 tables possibles
void free_huffman_trees(huffman_trees *tree){
    if (tree == NULL) return;
    for (int i = 0; i < 4; i++){
        if (tree->ac[i] !=NULL) free_huffman_tree(tree->ac[i]);
        if (tree->dc[i] !=NULL) free_huffman_tree(tree->dc[i]);
    }
    free(tree);
}
