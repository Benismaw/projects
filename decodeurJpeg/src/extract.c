
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include "extract.h"
#include "lecture_entete.h"
#include "zigzag.h"
#include "arbre.h"
#include "bitstream.h"


// On convertit une valeur codée en magnitude JPEG en entier signé
int16_t decode_magnitude(int code, int longueur){
    int msb = (code >> (longueur - 1)) & 1;
    uint16_t unsigned_code = (unsigned int) code;

    // Si le bit de poids fort est 1, la valeur est positive
    if (msb == 1) {
        return unsigned_code;
    }
    // Sinon, on fait le complément à 1 pour obtenir la valeur négative 
    uint16_t mask = (1U << longueur) - 1;
    int16_t complement1 = (~unsigned_code) & mask;
    return -complement1;
}

// On libère la mémoire allouée pour un bloc
void liberer_bloc(uint16_t *vecteur){
    if (vecteur != NULL) {
        free(vecteur);
    }
}

// On lit un symbole DC en parcourant l'arbre de Huffman bit par bit
uint8_t read_symbole_dc(parsed_file *pf, Node *abr_dc) {
    Node *current = abr_dc;
    while (current != NULL) {
        uint8_t bit = read_bit(pf);  // On lit un bit du flux

        // En fonction du bit, on avance dans l'arbre
        if (bit == 0) {
            current = current->fg; // aller au fils gauche
        } else {
            current = current->fd; // aller au fils droite 
        }

        // Si on atteint NULL, c'est que le flux est probablement corrompu
        if (current == NULL) {
            fprintf(stderr, "Error: Current == Null in Huffman tree traversal during read_symbole_dc.\n");
            exit(1);
        }

        // Quand on atteint une feuille, on retourne la valeur (magnitude)
        if (current->fg == NULL && current->fd == NULL) {
            return current->cle;
        }
    }

    fprintf(stderr, "Error in the Huffman tree\n");
    exit(1);
}

// On lit un symbole AC (run-length + magnitude) via Huffman
void read_symbole_ac(uint8_t code_ac[2], parsed_file *pf, Node *abr_ac) {
    Node *current = abr_ac;
    while (current != NULL) {
        uint8_t bit = read_bit(pf);

        // On descend dans l'arbre selon le bit
        if (bit == 0) {
            current = current->fg; // aller au fils gauche
        } else {
            current = current->fd; // aller au fils droite 
        }

        // Si on atteint NULL, c'est que le flux est probablement corrompu
        if (current == NULL) {
            fprintf(stderr, "Error: Current == Null in Huffman tree traversal during read_symbole_ac.\n");
            exit(1);
        }

        // Si on est sur une feuille, on décode le symbole
        if (current->fg == NULL && current->fd == NULL) {
            uint8_t symbole = current->cle;
            code_ac[0] = (symbole >> 4) & 0x0F;  // RLE
            code_ac[1] = symbole & 0x0F;         // magnitude
            break;
        }
    }
}

// On décode les coefficients DC et AC et on les stocke dans le vecteur 
void decodage_dc(uint16_t vect[64], parsed_file *pf, Node *abr_ac, Node *abr_dc) {
    if (pf == NULL) {
        printf("No pf available\n");
        return;
    }
    
    // On commence par lire le symbole DC via l'arbre de Huffman 
    uint8_t symbole_dc = read_symbole_dc(pf, abr_dc);
    uint8_t index = 0;
    int16_t valeur_dc;

    // Si le symbole vaut 0, alors le coefficient est 0
    if (symbole_dc == 0){
        valeur_dc = 0; 
    }
    else {
        // Sinon, on lit les bits nécessaires pour calculer la vraie valeur
        int16_t code = read_n_bits(pf, symbole_dc);
        valeur_dc = decode_magnitude(code,symbole_dc);
    }
    vect[index] = valeur_dc;// On place le DC en premier
    index++;
    
    // Ensuite, on lit les coefficients AC jusqu'à remplir les 64 cases
    while (index < 64) {
        uint8_t code_ac[2];
        read_symbole_ac(code_ac, pf, abr_ac);
        uint8_t nb_zero = code_ac[0];    // nombre de zéros à insérer (RUN)
        uint8_t magnitude = code_ac[1];  // magnitude

        // On insère les zéros correspondant à nb_zero (run-length)
        for (int j = 1; j <= nb_zero && index < 64; j++) {
            vect[index] = 0;
            index++;
        }

        // Si magnitude > 0, on lit et décode un coefficient AC
        if (magnitude > 0 && index < 64) {
            int16_t codage_ac = read_n_bits(pf, magnitude);
            int16_t valeur_ac = decode_magnitude(codage_ac,magnitude);
            vect[index++] = valeur_ac;
        }

        // End of block (EOB)
        if (code_ac[0] == 0 && code_ac[1] == 0) {
            break;
        }

        // cas spécial: 0xF0
        if (code_ac[0] == 0xF  && code_ac[1] == 0) {
            vect[index++] = 0;
        }
    }

    // on complète le bloc avec des zéros si besoin 
    while (index < 64) {
        vect[index++] = 0;
    }
}
// On extrait un bloc de coefficients pour chaque MCU
uint16_t *extract_bloc(parsed_file *pf, huffman_trees* trees, int idx_dc, int idx_ac){
    // On alloue l'espace pour stocker 64 coefficients (8x8)
    uint16_t *vecteur = malloc(sizeof(uint16_t)*64);
    if (vecteur == NULL) {
        fprintf(stderr, "Memory allocation error for vecteur.\n");
        exit(1);
    }
    // On récupère les bons arbres de Huffman pour DC et AC
    Node *abr_huff_dc = trees->dc[idx_dc];
    Node *abr_huff_ac = trees->ac[idx_ac];

    // On décode les coefficients et on les place dans le vecteur
    decodage_dc(vecteur,pf,abr_huff_ac,abr_huff_dc);

    return vecteur;
}





