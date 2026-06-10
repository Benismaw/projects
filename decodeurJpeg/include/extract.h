#ifndef EXTRACT_H
#define EXTRACT_H

#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>
#include "bitstream.h"
#include "arbre.h"
#include "quantif.h"
#include "zigzag.h"
#include "idct.h"
#include "lecture_entete.h"

// On libère la mémoire allouée pour un bloc
void liberer_bloc(uint16_t *vecteur);

// On extrait un bloc de coefficients pour chaque MCU
uint16_t *extract_bloc(parsed_file *pf, huffman_trees* trees, int idx_dc, int idx_ac);

#endif
