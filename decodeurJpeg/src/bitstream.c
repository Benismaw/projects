#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include "bitstream.h"

// Fonction pour lire 1 bit du flux binaire (bitstream)
uint8_t read_bit(parsed_file *pf){
    bitstream *flux = get_image_data(pf);
    if (flux->position_octet >= flux->taille) {
        fprintf(stderr, "Error: Attempt to read beyond the end of the bitstream\n");
        exit(1);
    }
    uint8_t bit = (flux->data[flux->position_octet] >> (7 - (flux->position_bit % 8))) & 0x01;
    flux->position_bit++;
    if (flux->position_bit == 8) {
        flux->position_bit = 0;
        flux->position_octet++;
    }
    return bit;
}

// Fonction pour lire n bits (entre 1 et 16) du bitstream
uint16_t read_n_bits(parsed_file *pf, uint8_t n){
    if (n < 1 || n > 16) {
        fprintf(stderr, "Error: Attempted to read %d bits; valid range is 1-16.\n", n);
        exit(1);
    }
    uint16_t result = 0;
    for (uint8_t i = 0; i < n; i++) {
        result = (result << 1) | read_bit(pf);
    }
    return result;
}

// Retourne la position absolue actuelle dans le flux (en bits)
size_t bitstream_position(parsed_file *pf){
    bitstream *flux = get_image_data(pf);
    return (flux->position_octet * 8) + flux->position_bit;
}
