#include <stdio.h>
#include <stdint.h>
#include "lecture_entete.h"

// QuantizationTable *qt; // table de quantification
void quantification_inv(QuantizationTable *q, uint16_t *vecteur, uint16_t *output_vecteur) {
    uint8_t *qt=q->table;
    for (int i = 0; i < 64; i++) {
        output_vecteur[i] = vecteur[i] * qt[i];
    }
}