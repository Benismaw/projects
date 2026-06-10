#ifndef __BITSTREAM_H__
#define __BITSTREAM_H__

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include "lecture_entete.h"

typedef struct parsed_file parsed_file;

// Structure représentant les données brutes de l’image
typedef struct bitstream{
    uint8_t *data;          // Le tableau de données (tampon des octets) (de taille 'taille')
    size_t taille;          // Taille totale des données
    size_t position_octet;  // Position actuelle dans le tampon (en octets) (0 - taille-1)
    size_t position_bit;    // Position actuelle dans l'octet actuel (en bits) (0-7)
} bitstream;

// Fonction pour lire 1 bit du flux binaire (bitstream)
uint8_t read_bit(parsed_file *pf);

// Fonction pour lire n bits (entre 1 et 16) du bitstream
uint16_t read_n_bits(parsed_file *pf, uint8_t n);

// Retourne la position absolue actuelle dans le flux (en bits)
size_t bitstream_position(parsed_file *pf);

#endif