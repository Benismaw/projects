#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "lecture_entete.h"
#include "arbre.h"
#include "extract.h"
#include "bitstream.h"
#include "zigzag.h"
#include "ecriture.h"
#include "conversion.h"
#include "mcu.h"
#include "upsample.h"
#include "fast_idct.h"

int print_usage_and_exit(const char *prog_name) {
    fprintf(stderr, "Usage: %s [-v][-h] <jpeg_file>\n", prog_name);
    return EXIT_FAILURE;
}

bool verbose = false;           // -v
bool huffman_verbose = false;   // -h
char *outfile = NULL;           // --outfile=<filename>
char output_filename[1024];

int main(int argc, char **argv) {
    /* Gestion des arguments */
    if (argc < 2 || argc > 5) return print_usage_and_exit(argv[0]);
    int file_index = 1;
    for (int i = 1; i < argc; i++) { // for each argument
        if (argv[i][0] == '-') {
            if (strncmp(argv[i], "--outfile=", 10) == 0) {
                outfile = argv[i] + 10; // Points to the filename after '='
            } else {
                for (int j = 1; argv[i][j] != '\0'; j++) {
                    if (argv[i][j] == 'v') verbose = true;
                    else if (argv[i][j] == 'h') huffman_verbose = true;
                    else return print_usage_and_exit(argv[0]);
                }
            }
        } else file_index = i;
    }
    if (file_index == argc || argv[file_index][0] == '-') return print_usage_and_exit(argv[0]);

// Lecture Entête
parsed_file *pf;
pf = parse_file(argv[file_index]);

uint8_t nb_components = get_nb_components_sof(pf);

if (outfile == NULL) { // if no --outfile option
    get_ppm_filename(argv[file_index], output_filename, nb_components);
    outfile = output_filename;
}

uint16_t width, height;
get_image_size(pf, &width, &height);
write_image_header(outfile, width, height, nb_components);

uint16_t predicateur[nb_components];
for (int i = 0; i < nb_components; i++) predicateur[i] = 0;

uint8_t h1, v1, h2 = 0, v2 = 0, h3 = 0, v3 = 0;
get_sampling_factor_sof(pf, 0, &h1, &v1);
if (nb_components == 3) {
    get_sampling_factor_sof(pf, 1, &h2, &v2);
    get_sampling_factor_sof(pf, 2, &h3, &v3);
}
uint8_t nb_blocs_Y = h1 * v1, nb_blocs_Cb = h2 * v2, nb_blocs_Cr = h3 * v3;
uint8_t blocks_per_mcu = nb_blocs_Y + nb_blocs_Cb + nb_blocs_Cr;

int mcu_par_ligne = (width  + 8*h1 - 1) / (8*h1);

huffman_trees* trees = get_dht(pf);

uint8_t comp = 0; // 0 = Y, 1 = Cb, 2 = Cr

MCU mcu_line[mcu_par_ligne]; // array de mcu d'une ligne

int idx_mcu = 0, nb_mcu = 0;

size_t taille = get_image_data(pf)->taille;
bitstream *flux = get_image_data(pf);

while (taille - 1 > (flux->position_octet)) {
    for (int k = 0; k < blocks_per_mcu; k++) {
        if (k < nb_blocs_Y)
            comp = 0; // Y
        else if (k < nb_blocs_Y + nb_blocs_Cb)
            comp = 1; // Cb
        else
            comp = 2; // Cr

        // Extraction blocs
        uint8_t idx_dc, idx_ac;
        get_huffman_table_id_sos(pf, comp, &idx_dc, &idx_ac);
        uint16_t *vecteur = extract_bloc(pf, trees, idx_dc, idx_ac);
        vecteur[0] += predicateur[comp];
        predicateur[comp] = vecteur[0];

        // Quantification inverse
        uint16_t vecteur_quant[64];
        uint8_t idx_q = get_quantization_table_iq_sof(pf, comp);
        QuantizationTable *q = get_dqt(pf, idx_q);
        quantification_inv(q, vecteur, vecteur_quant);

        // Inverse Zig-zag
        int16_t matrix[8][8];
        zigzag(vecteur_quant, matrix);

        // Fast iDCT
        if (comp == 0) {
            fast_idct(mcu_line[idx_mcu].Y[k], matrix);
        }
        else if (comp == 1) {
            fast_idct(mcu_line[idx_mcu].Cb[k - nb_blocs_Y], matrix);

        }
        else {
            fast_idct(mcu_line[idx_mcu].Cr[k - nb_blocs_Y - nb_blocs_Cb], matrix);
        }

        liberer_bloc(vecteur);
    }

    // Upsampling
    if ((nb_components == 3) && (h1 != h2 || v1 != v2 || h1 != h3 || v1 != v3)){
        upsampling(&mcu_line[idx_mcu],h1,v1,h2,v2,h3,v3);
    }

    // Conversion YCbCr vers RGB
    if (nb_components == 3) conversion_rgb(&mcu_line[idx_mcu], h1, v1);

    idx_mcu ++;
    nb_mcu ++;
    // Ecriture du fichier PPM (ou PGM) */
    if (idx_mcu == mcu_par_ligne) { 
        write_mcu_line(outfile, mcu_line, nb_components, width, height, nb_mcu, h1, v1);
    }
    idx_mcu = idx_mcu % (mcu_par_ligne);
    }
    free_parsed_file(pf);
    return EXIT_SUCCESS;
}