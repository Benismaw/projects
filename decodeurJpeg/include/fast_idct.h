#ifndef FAST_IDCT_H
#define FAST_IDCT_H

#include <stdint.h>

/**
 * Performs the Loeffler algorithm for fast IDCT
 * This is an implementation of the inverse discrete cosine transform
 * using the Loeffler-Ligtenberg-Moschytz algorithm
 *
 * @param s Output 8x8 block of pixel values (range 0-255)
 * @param matrix Input 8x8 block of DCT coefficients
 */
void fast_idct(uint8_t s[8][8], int16_t matrix[8][8]);

#endif