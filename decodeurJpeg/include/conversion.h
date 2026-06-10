#ifndef CONVERSION_H
#define CONVERSION_H
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "mcu.h"

// Convertit chaque pixel YCbCr d'un MCU en RGB et stocke le résultat dans les mêmes blocs.
void conversion_rgb(MCU *mcu, uint8_t h1, uint8_t v1);

#endif 