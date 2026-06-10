#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "ecriture.h"

// Génère le nom du fichier de sortie (.pgm pour NB ou .ppm pour couleur)
void get_ppm_filename(const char *jpeg_nomfichier, char *ppm_nomfichier, int nb_comp) {
    strcpy(ppm_nomfichier, jpeg_nomfichier); // Copier le nom original
    char *point = strrchr(ppm_nomfichier, '.'); // Trouver le dernier point (avant l'extension)

    if (point != NULL) {
        if(nb_comp==1){
            strcpy(point, ".pgm"); // Remplacer l'extension JPEG ou JPG par PGM
        }
        else{
            strcpy(point, ".ppm");
        }
    }
}

// Écrire l'entête du fichier PPM (P6) ou PGM (P5)
void write_image_header(const char *filename, int width, int height, int nb_comp){
    FILE *fp = fopen(filename, "wb");
    if (!fp){
        perror("Erreur dans l'ouverture de l'entête");
        return;
    }
    if (nb_comp == 1){
        fprintf(fp, "P5\n%d %d\n255\n", width, height); // En-tête du fichier PGM
    }
    else{
        fprintf(fp, "P6\n%d %d\n255\n", width, height); // En-tête du fichier PPM
    }
    fclose(fp);
}

// Vérifie si la ligne de pixel actuelle dépasse la hauteur de l’image
int is_truncated_bottom(int nb_mcu, int mcu_per_line, int v1, int line_in_block, int v_index, int image_height) {
    int current_pixel_y = (nb_mcu / mcu_per_line) * v1 * 8 + v_index * 8 + line_in_block - 8 * v1;
    return current_pixel_y >= image_height;
}

// Calcule le nombre de pixels à écrire pour une ligne de bloc 8x8
int get_write_size(int width, int m, int h, int h1) {
    // Calculer la colonne du pixel de départ pour ce bloc
    int x = (m * h1 + h) * 8;
    
    if (x + 8 > width) {
        // Retourner uniquement les pixels visibles dans ce bloc
        return width - x;
    }
    return 8;
}

// Écrit une ligne de blocs MCU décodés dans un fichier PPM,
// en gérant les composantes couleur et la troncature aux bords de l’image
void write_mcu_line(const char *filename, MCU *mcu_line, uint8_t nb_comp, uint16_t width, uint16_t height, int nb_mcu, uint8_t h1, uint8_t v1) {
    int mcu_par_ligne = (width  + 8*h1 - 1) / (8*h1);
    uint8_t vect_a_ecrire[nb_comp * 8]; // RGB or Y
    size_t write_size = 8;

    FILE *fp = fopen(filename, "ab"); // Ouverture du fichier en mode binaire
    if (!fp) {
        perror("Erreur dans l'ouverture du fichier PGM");
        return;
    }

    for (int v = 0; v < v1; v++) {
        for (int line = 0; line < 8; line ++) {
            if (is_truncated_bottom(nb_mcu, mcu_par_ligne, v1, line, v, height)) {
                fclose(fp);
                return;
            }
            for (int m = 0; m < mcu_par_ligne; m++) {
                for (int h = 0; h < h1; h++) {
                    // Ignorer ce bloc 8×8 si sa colonne de départ dépasse la largeur de l’image
                    if ((m * h1 + h) * 8 >= width) continue;
                    uint8_t idx = 0;
                    for (int column = 0; column < 8; column++){
                        vect_a_ecrire[idx++] = mcu_line[m].Y[h+v*h1][line][column]; // Y or R
                        if (nb_comp == 3){
                            vect_a_ecrire[idx++] = mcu_line[m].Cb[h+v*h1][line][column]; // G
                            vect_a_ecrire[idx++] = mcu_line[m].Cr[h+v*h1][line][column]; // B
                        }
                    }
                    write_size = get_write_size(width, m, h, h1); // gestion du troncature à droite
                    fwrite(vect_a_ecrire, 1, write_size*nb_comp, fp);
                }
            }
        }
    }
    fclose(fp);
}
