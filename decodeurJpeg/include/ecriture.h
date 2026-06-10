#ifndef PPM_SIMPLIFIE_H
#define PPM_SIMPLIFIE_H

#include "mcu.h"

// Génère le nom du fichier de sortie (.pgm pour NB ou .ppm pour couleur)
void get_ppm_filename(const char *jpeg_nomfichier, char *ppm_nomfichier, int nb_comp);

// Écrire l'entête du fichier PPM (P6) ou PGM (P5)
void write_image_header(const char *filename, int width, int height, int nb_comp);

// Écrit une ligne de blocs MCU décodés dans un fichier PPM,
// en gérant les composantes couleur et la troncature aux bords de l’image
void write_mcu_line(const char *filename, MCU *mcu_line, uint8_t nb_comp, uint16_t width, uint16_t height, int nb_mcu, uint8_t h1, uint8_t v1);

#endif // PPM_SIMPLIFIE_H