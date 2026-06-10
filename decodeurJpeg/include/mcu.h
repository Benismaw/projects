#ifndef MCU_H
#define MCU_H

#include <stdint.h>

typedef uint8_t bloc[8][8];

// Maximum number of blocks per component in an MCU:
// - Baseline sequential JPEG: max 6 blocks for Y (h1=3, v1=2)
// - Baseline progressive JPEG: max 16 blocks for Y (h1=4, v1=4)
typedef struct {
    bloc Y[16];
    bloc Cb[16];
    bloc Cr[16];
} MCU;

#endif