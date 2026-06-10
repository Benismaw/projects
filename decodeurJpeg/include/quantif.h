#ifndef QUANTIF_H
#define QUANTIF_H

#include <stdint.h>
#include <stdio.h>
#include "lecture_entete.h"

void quantification_inv(QuantizationTable *q, uint16_t *vecteur, uint16_t *output_vecteur);

#endif 