#ifndef IDCT_H    // Si pas encore défini
#define IDCT_H    // On le définit

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include <math.h>

#include "mcu.h"

// Fonction IDCT (Inverse Discrete Cosine Transform)
void idct(bloc s, int16_t matrix[8][8]);

#endif 
