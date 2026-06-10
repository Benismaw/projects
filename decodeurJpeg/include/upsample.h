#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "mcu.h"


// Fonction principale qui applique le bon type d'up-sampling selon le format de sous-échantillonnage
void upsampling(MCU *mcu ,uint8_t h1,uint8_t v1,uint8_t h2,uint8_t v2,uint8_t h3,uint8_t v3);