#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "idct.h"
#include <math.h>

#ifndef PI
#define PI 3.14159265358979323846
#endif 

// Fonction de normalisation utilisée dans l’IDCT
float fct(int a) {
    if (a != 0) {
        return 1.0;
    } else {
        return (1 / sqrt(2));
    }
}
// Fonction IDCT (Inverse Discrete Cosine Transform)
void idct(bloc s, int16_t matrix[8][8]) {
    float cos_table[8][8]; 
    for (int x = 0; x < 8; x++) {
        for (int y = 0; y < 8; y++) {
            cos_table[x][y]= cos((2 * x + 1) * y * PI / 16);
        }
    }
    for (int x = 0; x < 8; x++) {
        for (int y = 0; y < 8; y++) {
            float somme_l = 0;
            for (int l = 0; l < 8; l++) {
                float somme_u = 0;
                for (int u = 0; u < 8; u++) {
                    // facteurs de normalisation 
                    float fact1, fact2;
                    fact1 = fct(l);
                    fact2 = fct(u);
                    int elem = matrix[l][u];
                    somme_u += fact1 * fact2 * cos_table[x][l] * cos_table[y][u] * elem;
                }
                somme_l += somme_u;
            }

            // Application du facteur final (1/4 selon la formule IDCT)
            float val = 0.25 * somme_l;
            val += 128.0;
            if (val < 0) {
                val = 0;
            }
            if (val > 255) {
                val = 255;
            }

            // Arrondi final
            s[x][y] = (uint8_t)round(val);
        }
    }
}


//     return EXIT_SUCCESS;
// }
