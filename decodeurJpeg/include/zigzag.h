#ifndef ZIGZAG_H
#define ZIGZAG_H

#include <stdint.h>

//Convertit un vecteur 1D de coefficients (longueur 64) en une matrice 2D 8x8
void zigzag(uint16_t *vect, int16_t matrix[8][8]);

#endif 

