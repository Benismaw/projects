#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "conversion.h"

// Applique une saturation à la valeur : ramène val dans [0,255].
uint8_t saturation(float val){
    if (val < 0.0) return 0;
    if (val > 255.0) return 255;
    return (uint8_t)val;
}

// Convertit chaque pixel YCbCr d'un MCU en RGB et stocke le résultat dans les mêmes blocs.
void conversion_rgb(MCU *mcu, uint8_t h1, uint8_t v1) {
    if (mcu == NULL) {
        fprintf(stderr, "MCU is NULL in conversion RGB!");
        exit(1);
    }

    int bloc_par_comp = h1*v1; // Nb de blocs Y (et donc de blocs Cb/Cr après upsampling)

    // Conversion pixel par pixel pour chaque bloc de l'MCU
    for (int k = 0; k < bloc_par_comp; k++) {
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                uint8_t Y  = mcu->Y[k][i][j];
                uint8_t Cb = mcu->Cb[k][i][j];
                uint8_t Cr = mcu->Cr[k][i][j];

                // Formules de conversion YCbCr -> RGB
                float R = Y + 1.402f * (Cr - 128);
                float G = Y - 0.34414f * (Cb - 128) - 0.71414f * (Cr - 128);
                float B = Y + 1.772f * (Cb - 128);

                // Saturation et stockage dans les blocs respectifs
                mcu->Y[k][i][j] = saturation(R); // R
                mcu->Cb[k][i][j] = saturation(G); // G
                mcu->Cr[k][i][j] = saturation(B); // B
            }
        }
    }
}
