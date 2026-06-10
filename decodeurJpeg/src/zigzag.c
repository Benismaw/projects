#include <stdio.h>
#include <zigzag.h>

//Convertit un vecteur 1D de coefficients (longueur 64) en une matrice 2D 8x8
void zigzag(uint16_t *vect, int16_t matrix[8][8]){
    // Ordre inverse du parcours zigzag (chaque paire représente les indices ligne/colonne)
    int16_t ord_zigzag[64][2]={{0,0}, {0,1}, {1,0}, {2,0}, {1,1}, {0,2}, {0,3}, {1,2},
    {2,1}, {3,0}, {4,0}, {3,1}, {2,2}, {1,3}, {0,4}, {0,5},
    {1,4}, {2,3}, {3,2}, {4,1}, {5,0}, {6,0}, {5,1}, {4,2},
    {3,3}, {2,4}, {1,5}, {0,6}, {0,7}, {1,6}, {2,5}, {3,4},
    {4,3}, {5,2}, {6,1}, {7,0}, {7,1}, {6,2}, {5,3}, {4,4},
    {3,5}, {2,6}, {1,7}, {2,7}, {3,6}, {4,5}, {5,4}, {6,3},
    {7,2}, {7,3}, {6,4}, {5,5}, {4,6}, {3,7}, {4,7}, {5,6},
    {6,5}, {7,4}, {7,5}, {6,6}, {5,7}, {6,7}, {7,6}, {7,7}
    };
    for (size_t i=0; i<64; i++){
        int16_t ligne = ord_zigzag[i][0];
        int16_t col = ord_zigzag[i][1];
        matrix[ligne][col] = vect[i];
    }
}
