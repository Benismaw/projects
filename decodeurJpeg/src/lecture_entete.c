#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "lecture_entete.h"


extern bool verbose; // -v
extern bool huffman_verbose;   // -h

/* Structure representing a parsed JPEG file */
struct parsed_file {
    bool soi_marker; // Start of Image marker (0xFFD8, exactly one, mandatory)
    struct sof_segment* sof; // Start of Frame (0xFFC0/2, exactly one)
    QuantizationTable *tables[4]; // Define Quantization Table (0xFFDB, usually 2, max 4 tables)
    huffman_trees* trees; // All Huffman Trees (DC and AC)
    struct sos_segment* sos; // Start of Scan (0xFFDA, one in sequential JPEG)
    bitstream* image_data; // Compressed image data (between SOS and EOI markers)
    bool progressive; // True if the JPEG is progressive (not supported yet)
    bool eoi_marker; // End of Image marker (0xFFD9, exactly one, mandatory)
};

struct sof_segment {
    uint8_t precision; // always 8
    uint16_t height;
    uint16_t width;
    uint8_t nb_components; // 1 gris, 3 YCbCr
    struct sof_component *components;
};

struct sof_component { // for each Y, Cb and Cr component
    uint8_t ic; // de 0 à 255
    uint8_t horizontal_sampling_factor; // 1 à 4 (4 bits)
    uint8_t vertical_sampling_factor; // 1 à 4 (4 bits)
    uint8_t quantization_table_id; // 0-3 (iq)
};

struct sos_segment {
    uint8_t nb_components;
    struct sos_component* components; // array of components
    uint8_t start_of_selection; // 0 in baseline
    uint8_t end_of_selection; // 63 in baseline
    uint8_t successive_approximation; //0b 0000 0000 in baseline
};

struct sos_component {
    uint8_t ic;
    uint8_t dc_huffman_table_id; // 0-3 (dc) (4 bits)
    uint8_t ac_huffman_table_id; // 0-3 (ac) (4 bits)
};

void exit_with_error(parsed_file *pf, FILE *f) {
    if (f != NULL) fclose(f);
    if (pf != NULL) free_parsed_file(pf);
    exit(1);
}

size_t safe_fread(void *buffer, size_t size, size_t count, FILE *f, const char *component, parsed_file *pf) {
    size_t bytesRead = fread(buffer, size, count, f);
    if (bytesRead != count) {
        fprintf(stderr, "Error: Expected %ld bytes but read %ld for %s\n", count, bytesRead, component);
        exit_with_error(pf, f);
    }
    return bytesRead;
}

void safe_fseek(FILE *f, long offset, int whence, const char *component, parsed_file *pf) {
    if (fseek(f, offset, whence) != 0) {
        perror(component);
        exit_with_error(pf, f);
    }
}

void *malloc_and_error_handling(size_t size, FILE *f, const char *component, parsed_file *pf) {
    void *ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for %s\n", component);
        perror(component);
        exit_with_error(pf, f);
    }
    return ptr;
}

parsed_file *new_parsed_file(){
    parsed_file *pf = malloc_and_error_handling(sizeof(parsed_file), NULL, "parsed_file structure", NULL);
    pf->soi_marker = false;
    for (int i = 0; i < 4; i++) {
        pf->tables[i] = NULL;
    }
    pf->sof = NULL;
    pf->trees = NULL;
    pf->sos = NULL;
    pf->image_data = NULL;
    pf->progressive = false;
    pf->eoi_marker = false;
    return pf;
}

void parse_app0(parsed_file *pf, FILE *f) {
    uint8_t buffer[2];
    safe_fread(buffer, 1, sizeof(buffer), f, "APP0 length", pf);
    uint16_t length = (buffer[0] << 8) | buffer[1];
    if (verbose) printf("[APP0]  length %d bytes\n", length); // -v
    uint8_t buffer5[5];
    safe_fread(buffer5, 1, sizeof(buffer5), f, "APP0 identifier", pf);
    if (buffer5[0] != 'J' || buffer5[1] != 'F' || buffer5[2] != 'I' 
        || buffer5[3] != 'F' || buffer5[4] != '\0') {
        fprintf(stderr, "Invalid identifier: %s != JFIF\\0\n", buffer5);
        exit_with_error(pf, f);
    }
    safe_fread(buffer, 1, sizeof(buffer), f, "APP0 version", pf);
    if (buffer[0] != 1 || buffer[1] != 1) {
        fprintf(stderr, "Invalid JFIF version: %d.%02d (expected 1.01)\n", buffer[0], buffer[1]);
        exit_with_error(pf, f);
    }
    if (verbose) printf("\tJFIF application version %d.%02d\n", buffer[0], buffer[1]); // -v
    safe_fseek(f, 7, SEEK_CUR, "APP0 skip 7 bytes", pf);
    if (verbose) printf("\tother parameters ignored (7 bytes).\n"); // -v
}

void parse_com(parsed_file *pf, FILE *f) {
    uint8_t buffer[2];
    safe_fread(buffer, 1, sizeof(buffer), f, "COM length", pf);
    uint16_t length = (buffer[0] << 8) | buffer[1];
    if (verbose) printf("[COM]   length %d bytes\n", length); // -v
    char *comment = malloc_and_error_handling(length - 1, f, "COM comment buffer", pf);
    safe_fread(comment, 1, length - 2, f, "COM comment",pf);
    comment[length - 2] = '\0'; // Ensure null-terminated
    if (verbose) printf("\tComment found: \"%s\"\n", comment);
    free(comment);
}

void parse_dqt(parsed_file *pf, FILE *f) {
    uint8_t buffer[2];
    safe_fread(buffer, 1, sizeof(buffer), f, "DQT length", pf);
    uint16_t length = (buffer[0] << 8) | buffer[1];
    if (verbose) printf("[DQT]   length %d bytes\n", length); // -v
    size_t remaining = length - 2;

    while (remaining > 0){
        QuantizationTable *new_table = malloc_and_error_handling(sizeof(QuantizationTable), f, "Quantization Table", pf);
        uint8_t buffer1[1];
        safe_fread(buffer1, 1, sizeof(buffer1), f, "DQT precision & iq", pf);
        remaining -= 1;
        new_table->precision = (buffer1[0] >> 4); // 4 bits
        new_table->iq = buffer1[0] & 0x0F; // 4 bits
        if (new_table->iq >= 4) { // 4 bits for index
            fprintf(stderr, "Invalid DQT table index: %d\n", new_table->iq);
            exit_with_error(pf, f);
        }
        if (verbose) printf("\t\tquantization table index %d\n", new_table->iq); // -v
        if (verbose) printf("\t\tquantization precision %d bits\n", new_table->precision == 0 ? 8 : 16); // -v
        for (size_t i = 0; i < 64; i++) {
            if (new_table->precision == 0) { // 8‑bit value
                safe_fread(buffer1, 1, 1, f, "DQT table 8-bit", pf);
                new_table->table[i] = buffer1[0];
                remaining -= 1;
            } else if (new_table->precision == 1) { // 16‑bit value
                fprintf(stderr, "16-bit precision not supported yet\n");
                exit_with_error(pf, f);
            } else {
                fprintf(stderr, "Invalid precision: %d\n", new_table->precision);
                exit_with_error(pf, f);
            }
        }
        if (verbose) printf("\t\tquantization table read (%d bytes)\n", new_table->precision == 0 ? 64 : 128); // -v
        pf->tables[new_table->iq] = new_table;
    }
}

void parse_sof(parsed_file *pf, FILE *f, int sof_type) { // sof0 or sof2
    pf->sof = malloc_and_error_handling(sizeof(struct sof_segment), f, "SOF segment structure", pf);
    uint8_t buffer[2];
    safe_fread(buffer, 1, sizeof(buffer), f, "SOF length", pf);
    uint16_t length = (buffer[0] << 8) | buffer[1];
    if (verbose) printf("[SOF%d]  length %d bytes\n", sof_type, length); // -v
    uint8_t buffer1[1];
    safe_fread(buffer1, 1, sizeof(buffer1), f, "SOF precision", pf);
    pf->sof->precision = buffer1[0];
    if (pf->sof->precision != 8) {
        fprintf(stderr, "Unsupported precision: %d (only 8 bits supported)\n", pf->sof->precision);
        exit_with_error(pf, f);
    }
    if (verbose) printf("\tsample precision %d\n", pf->sof->precision); // -v
    safe_fread(buffer, 1, sizeof(buffer), f, "SOF height", pf);
    pf->sof->height = (buffer[0] << 8) | buffer[1];
    if (verbose) printf("\timage height %d\n", pf->sof->height); // -v
    safe_fread(buffer, 1, sizeof(buffer), f, "SOF width", pf);
    pf->sof->width = (buffer[0] << 8) | buffer[1];
    if (verbose) printf("\timage width %d\n", pf->sof->width); // -v
    safe_fread(buffer1, 1, sizeof(buffer1), f, "SOF nb_components", pf);
    pf->sof->nb_components = buffer1[0];
    if (verbose) printf("\tnb of component %d\n", pf->sof->nb_components); // -v
    pf->sof->components = malloc_and_error_handling(pf->sof->nb_components * sizeof(struct sof_component), f, "SOF components array", pf);
    int sum_sampling_factors = 0; // 
    for (size_t i = 0; i < pf->sof->nb_components; i++)
    {
        struct sof_component* component = &(pf->sof->components[i]);
        if (verbose) printf("\tcomponent %s\n", (i==0) ? "Y" : ((i==1) ? "Cb" : "Cr")); // -v
        safe_fread(buffer1, 1, sizeof(buffer1), f, "SOF component ic", pf);
        component->ic = buffer1[0];
        if (verbose) printf("\t\tid %d\n", component->ic); // -v
        safe_fread(buffer1, 1, sizeof(buffer1), f, "SOF component sampling_factor", pf);
        component->horizontal_sampling_factor = (buffer1[0] >> 4) & 0x0F; // 4 bits
        component->vertical_sampling_factor = buffer1[0] & 0x0F; // 4 bits
        sum_sampling_factors+= component->horizontal_sampling_factor * component->vertical_sampling_factor;
        if (component->horizontal_sampling_factor > 4 || component->vertical_sampling_factor > 4) {
            fprintf(stderr, "Invalid sampling factors: %d x %d (should be <= 4)\n", component->horizontal_sampling_factor, component->vertical_sampling_factor);
            exit_with_error(pf, f);
        }
        if (i > 0) {
            struct sof_component* luminance = &(pf->sof->components[0]);
            if (luminance->horizontal_sampling_factor % component->horizontal_sampling_factor != 0 ||
                luminance->vertical_sampling_factor % component->vertical_sampling_factor != 0) {
                fprintf(stderr, "Chrominance sampling factors must perfectly divide those of luminance (Y): %dx%d vs %dx%d\n",
                        component->horizontal_sampling_factor, component->vertical_sampling_factor,
                        luminance->horizontal_sampling_factor, luminance->vertical_sampling_factor);
                exit_with_error(pf, f);
            }
        }
        if (verbose) printf("\t\tsampling factors (hxv) %dx%d\n", component->horizontal_sampling_factor, component->vertical_sampling_factor); // -v
        safe_fread(buffer1, 1, sizeof(buffer1), f, "SOF component iq", pf);
        component->quantization_table_id = buffer1[0];
        if (verbose) printf("\t\tquantization table index %d\n", component->quantization_table_id); // -v
    }
    if (sum_sampling_factors > 10){
        fprintf(stderr, "Invalid sampling factors: %d (should be <= 10)\n", sum_sampling_factors);
        exit_with_error(pf, f);
    }
}

void parse_dht(parsed_file *pf, FILE *f) {
    uint8_t buffer[2];
    safe_fread(buffer, 1, sizeof(buffer), f, "DHT length", pf);
    uint16_t length = ((buffer[0] << 8) | buffer[1]);
    if (verbose) printf("[DHT]   length %d bytes\n", length); // -v
    size_t remaining = length - 2;
    if (pf->trees == NULL) {
        pf->trees = malloc_and_error_handling(sizeof(huffman_trees), f, "DHT segment structure", pf);
        for (int i = 0; i < 4; i++) {
            pf->trees->ac[i] = NULL;
            pf->trees->dc[i] = NULL;
        }
    }
    uint8_t buffer1[1];
    while (remaining > 0){
        safe_fread(buffer1, 1, sizeof(buffer1), f, "DHT table info", pf);
        remaining -= 1;
        uint8_t dc_ac = (buffer1[0] >> 4) & 0x01; // 0 for DC, 1 for AC
        uint8_t index = buffer1[0] & 0x0F; // 4 bits
        if (((buffer1[0] >> 5) & 0x07) != 0) { // 3 bits must be 0
            fprintf(stderr, "Invalid DHT table info: %d\n", buffer1[0]);
            exit_with_error(pf, f);
        }
        if (dc_ac != 0 && dc_ac != 1) { // 0 for DC, 1 for AC
            fprintf(stderr, "Invalid DHT table type (DC_AC): %d\n", dc_ac);
            exit_with_error(pf, f);
        }
        if (index >= 4) { // 4 bits for index
            fprintf(stderr, "Invalid DHT table index: %d\n", index);
            exit_with_error(pf, f);
        }
        HuffmanTable htable;
        htable.dc_ac = dc_ac;
        htable.index = index;
        if (verbose) printf("\t\tHuffman table type %s\n", dc_ac == 0 ? "DC" : "AC"); // -v
        if (verbose) printf("\t\tHuffman table index %d\n", index); // -v
        uint16_t sum_offsets = 0;
        for (size_t i = 0; i < 16; i++) {
            safe_fread(buffer1, 1, sizeof(buffer1), f, "DHT table offsets", pf);
            remaining -= 1;
            htable.offsets[i] = buffer1[0];
            sum_offsets += buffer1[0];
        }
        htable.sum_offsets = sum_offsets;
        if (verbose) printf("\t\ttotal nb of Huffman symbols %d\n",sum_offsets); // -v
        htable.valeurs = malloc_and_error_handling(sum_offsets * sizeof(uint8_t), f, "DHT table values", pf);
        for (size_t i = 0; i < sum_offsets; i++) {
            safe_fread(buffer1, 1, sizeof(buffer1), f, "DHT table values", pf);
            remaining -= 1;
            htable.valeurs[i] = buffer1[0];
        }
        if (dc_ac == 0) {
            if (pf->trees->dc[index] != NULL) {
                free_huffman_tree(pf->trees->dc[index]);
            }
            pf->trees->dc[index] = decode_huff(&htable);
            if (verbose && huffman_verbose) print_huffman_paths_bfs(pf->trees->dc[index]); // -h
        } else {
            if (pf->trees->ac[index] != NULL) {
                free_huffman_tree(pf->trees->ac[index]);
            }
            pf->trees->ac[index] = decode_huff(&htable);
            if (verbose && huffman_verbose) print_huffman_paths_bfs(pf->trees->ac[index]); // -h
        }
        free(htable.valeurs);
    }
}

void parse_markers(parsed_file *pf, FILE *f, uint8_t marker[2]);

void parse_image_data(parsed_file *pf, FILE *f) {
    if (verbose) printf("... (parsing image data) ...\n"); // -v
    size_t buffer_size = 1024; // Taille initiale du tampon
    size_t image_data_capacity = buffer_size;
    pf->image_data = malloc_and_error_handling(image_data_capacity, f, "Image data", pf);
    pf->image_data->position_bit = 0;
    pf->image_data->position_octet = 0;
    pf->image_data->data = malloc_and_error_handling(image_data_capacity, f, "Image data", pf);
    pf->image_data->taille = 0;
    uint8_t buffer1[1];
    while (true) {
        safe_fread(buffer1, 1, sizeof(buffer1), f, "Image data", pf);

        if (buffer1[0] == 0xFF) {
            safe_fread(buffer1, 1, sizeof(buffer1), f, "Image data after FF", pf);

            if (buffer1[0] == 0x00) { // byte stuffing
                buffer1[0] = 0xFF;
            } else {
                uint8_t marker[2] = {0xFF, buffer1[0]};
                parse_markers(pf, f, marker);
                if (pf->eoi_marker) {
                    break; // Arrêter si le marqueur EOI est trouvé
                }
                continue; // Ignorer les autres marqueurs
            }
        }

        // Ajouter l'octet aux données d'image
        if (pf->image_data->taille >= image_data_capacity) {
            // Augmenter la capacité si nécessaire
            image_data_capacity *= 2;
            pf->image_data->data = realloc(pf->image_data->data, image_data_capacity);
            if (pf->image_data->data == NULL) {
                fprintf(stderr, "Error: Failed to reallocate memory for image data\n");
                exit_with_error(pf, f);
            }
        }
        pf->image_data->data[pf->image_data->taille++] = buffer1[0];
    }

    if (verbose) printf("Image data read (%zu bytes)\n", pf->image_data->taille); // -v
}


void parse_sos(parsed_file *pf, FILE *f) {
    if (pf->sos != NULL) {
        if (pf->sos->components != NULL) free(pf->sos->components);
        free(pf->sos);
    }
    pf->sos = malloc_and_error_handling(sizeof(struct sos_segment), f, "SOS segment structure", pf);
    uint8_t buffer[2];
    safe_fread(buffer, 1, sizeof(buffer), f, "SOS length", pf);
    uint16_t length = (buffer[0] << 8) | buffer[1];
    if (verbose) printf("[SOS]   length %d bytes\n", length); // -v
    uint16_t remaining = length - 2;
    uint8_t buffer1[1];
    safe_fread(buffer1, 1, sizeof(buffer1), f, "SOS nb_components", pf);
    remaining -= 1;
    pf->sos->nb_components = buffer1[0];
    if (verbose) printf("\tnb of components in scan %d\n", pf->sos->nb_components); // -v
    pf->sos->components = malloc_and_error_handling(pf->sos->nb_components * sizeof(struct sos_component), f, "SOS components array", pf);
    for (size_t i = 0; i < pf->sos->nb_components; i++) {
        struct sos_component* component = &(pf->sos->components[i]);
        if (verbose) printf("\tscan component index %ld\n", i); // -v
        safe_fread(buffer1, 1, sizeof(buffer1), f, "SOS component ic", pf);
        component->ic = buffer1[0];
        if (verbose){ // Find frame index in SOF (pour l'affichage, useless in the actual parsing) // -v
            int frame_index = -1;
            for (size_t j = 0; j < pf->sof->nb_components; j++) {
                if (pf->sof->components[j].ic == component->ic) {
                    frame_index = j;
                    break;
                }
            }
            printf("\t\tassociated to component of id %d (frame index %d)\n", component->ic, frame_index); // -v
        }
        safe_fread(buffer1, 1, sizeof(buffer1), f, "SOS component dc_ac_huffman_table_id", pf);
        component->dc_huffman_table_id = (buffer1[0] >> 4) & 0x0F; // 4 bits
        if (verbose) printf("\t\tassociated to DC Huffman table of index %d\n", component->dc_huffman_table_id); // -v
        component->ac_huffman_table_id = buffer1[0] & 0x0F; // 4 bits
        if (verbose) printf("\t\tassociated to AC Huffman table of index %d\n", component->ac_huffman_table_id); // -v
        remaining -= 2;
    }
    safe_fread(buffer1, 1, sizeof(buffer1), f, "SOS start_of_selection", pf);
    pf->sos->start_of_selection = buffer1[0];
    if (verbose) printf("\tStart of selection (0 in baseline)  %d\n", pf->sos->start_of_selection); // -v
    safe_fread(buffer1, 1, sizeof(buffer1), f, "SOS end_of_selection", pf);
    pf->sos->end_of_selection = buffer1[0];
    if (verbose) printf("\tEnd of selection (63 in baseline)  %d\n", pf->sos->end_of_selection); // -v
    safe_fread(buffer1, 1, sizeof(buffer1), f, "SOS successive_approximation", pf);
    pf->sos->successive_approximation = buffer1[0];
    if (verbose) printf("\tSuccessive approximation (0 in baseline)  %d\n", pf->sos->successive_approximation); // -v
    remaining -= 3;
    if (verbose) printf("\tEnd of Scan Header (SOS)\n\n"); // -v
    parse_image_data(pf, f);
}

void parse_markers(parsed_file *pf, FILE *f, uint8_t marker[2]) {
    if (marker[0] != 0xFF) {
        fprintf(stderr, "Expected marker prefix 0xFF, got 0x%02X\n", marker[0]);
        exit_with_error(pf, f);
    }
    switch (marker[1]) {
        case 0xD8:
            if (verbose) printf("[SOI]   marker found\n"); // -v
            pf->soi_marker = true;
            break;
        case 0xE0:
            parse_app0(pf, f);
            break;
        case 0xFE:
            parse_com(pf, f);
            break;
        case 0xDB:
            parse_dqt(pf, f);
            break;
        case 0xC0:
            parse_sof(pf, f, 0);
            break;
            case 0xC4:
            parse_dht(pf, f);
            break;
            case 0xDA:
            parse_sos(pf, f);
            break;
        case 0xC2: // Progressive DCT
            // parse_sof(pf, f, 2);
        case 0xC6:
        case 0xCA:
        case 0xce:
        case 0xcf:
            pf->progressive = true;
            fprintf(stderr, "Marker 0xFF%02X is unsupported, progressive JPEGs are not supported yet.\n", marker[1]);
            exit_with_error(pf, f);
            break;
        case 0xD9:
            pf->eoi_marker = true;
            if (verbose) printf("[EOI]   marker found\n"); // -v
            break;
        default:
            fprintf(stderr,"Unsupported marker: 0xFF%02X\n",marker[1]);
            exit_with_error(pf, f);
            break;
    }
}

parsed_file *parse_file(char *name) {
    FILE *f = fopen(name, "rb");
    if (f == NULL) {
        perror("Failed to open file");
        exit_with_error(NULL, f);
    }

    parsed_file *pf = new_parsed_file();
    if (!pf) {
        perror("Failed to allocate memory for parsed_file");
        exit_with_error(pf, f);
    }

    uint8_t marker[2]; // 2 bytes not bits
    size_t bytesRead;

    while ((bytesRead = fread(marker, 1, sizeof(marker), f)) > 0) {
        if(bytesRead != 2){
            fprintf(stderr, "Bytes read = %ld\n Should've been 2 for the marker!\n", bytesRead);
            exit_with_error(pf, f);
        }
        parse_markers(pf, f, marker);
    }
    if (pf->soi_marker == false || pf->eoi_marker == false) {
        fprintf(stderr, "SOI or EOI marker not found\n");
        exit_with_error(pf, f);
    }
    if (pf->progressive) {
        fprintf(stderr, "Progressive JPEGs are not supported yet\n");
        exit_with_error(pf, f);
    }
    fclose(f);

    return pf;
}

/* SOI */
bool get_soi_marker(parsed_file *pf){
    return pf->soi_marker;
}

/* DQT */
QuantizationTable *get_dqt(parsed_file *pf, uint8_t index){
    if (index >= 4) {
        fprintf(stderr, "Invalid DQT index: %d\n", index);
        exit_with_error(pf, NULL);
    }
    return pf->tables[index];
}

/* SOF */
uint8_t get_precision(parsed_file *pf){
    if (pf->sof == NULL) {
        fprintf(stderr, "SOF segment not found\n");
        exit_with_error(pf, NULL);
    }
    return pf->sof->precision;
}

void get_image_size(parsed_file *pf, uint16_t *width, uint16_t *height){
    if (pf->sof == NULL) {
        fprintf(stderr, "SOF segment not found\n");
        exit_with_error(pf, NULL);
    }
    *width = pf->sof->width;
    *height = pf->sof->height;
}

uint8_t get_nb_components_sof(parsed_file *pf){
    if (pf->sof == NULL) {
        fprintf(stderr, "SOF segment not found\n");
        exit_with_error(pf, NULL);
    }
    return pf->sof->nb_components;
}
uint8_t get_component_ic_sof(parsed_file *pf, uint8_t index){
    if (pf->sof == NULL) {
        fprintf(stderr, "SOF segment not found\n");
        exit_with_error(pf, NULL);
    }
    if (index >= pf->sof->nb_components) {
        fprintf(stderr, "Invalid component index: %d\n", index);
        exit_with_error(pf, NULL);
    }
    return pf->sof->components[index].ic;
}

void get_sampling_factor_sof(parsed_file *pf, uint8_t index, uint8_t *horizontal, uint8_t *vertical){
    if (pf->sof == NULL) {
        fprintf(stderr, "SOF segment not found\n");
        exit_with_error(pf, NULL);
    }
    if (index >= pf->sof->nb_components) {
        fprintf(stderr, "Invalid component index: %d\n", index);
        exit_with_error(pf, NULL);
    }
    *horizontal = pf->sof->components[index].horizontal_sampling_factor;
    *vertical = pf->sof->components[index].vertical_sampling_factor;
}

uint8_t get_quantization_table_iq_sof(parsed_file *pf, uint8_t index){
    if (pf->sof == NULL) {
        fprintf(stderr, "SOF segment not found\n");
        exit_with_error(pf, NULL);
    }
    if (index >= pf->sof->nb_components) {
        fprintf(stderr, "Invalid component index: %d\n", index);
        exit_with_error(pf, NULL);
    }
    return pf->sof->components[index].quantization_table_id;
}

/* DHT */
huffman_trees *get_dht(parsed_file *pf){
    return pf->trees;
}

/* SOS */
uint8_t get_nb_components_sos(parsed_file *pf){
    if (pf->sos == NULL) {
        fprintf(stderr, "SOS segment not found\n");
        exit_with_error(pf, NULL);
    }
    return pf->sos->nb_components;
}

uint8_t get_component_ic_sos(parsed_file *pf, uint8_t index){
    if (pf->sos == NULL) {
        fprintf(stderr, "SOS segment not found\n");
        exit_with_error(pf, NULL);
    }
    if (index >= pf->sos->nb_components) {
        fprintf(stderr, "Invalid component index: %d\n", index);
        exit_with_error(pf, NULL);
    }
    return pf->sos->components[index].ic;
}

void get_huffman_table_id_sos(parsed_file *pf, uint8_t index, uint8_t *dc, uint8_t *ac){
    if (pf->sos == NULL) {
        fprintf(stderr, "SOS segment not found\n");
        exit_with_error(pf, NULL);
    }
    if (index >= pf->sos->nb_components) {
        fprintf(stderr, "Invalid component index: %d\n", index);
        exit_with_error(pf, NULL);
    }
    *dc = pf->sos->components[index].dc_huffman_table_id;
    *ac = pf->sos->components[index].ac_huffman_table_id;
}

void get_start_end_selection_sos(parsed_file *pf, uint8_t *start, uint8_t *end){
    if (pf->sos == NULL) {
        fprintf(stderr, "SOS segment not found\n");
        exit_with_error(pf, NULL);
    }
    *start = pf->sos->start_of_selection;
    *end = pf->sos->end_of_selection;
}

uint8_t get_successive_approximation_sos(parsed_file *pf){
    if (pf->sos == NULL) {
        fprintf(stderr, "SOS segment not found\n");
        exit_with_error(pf, NULL);
    }
    return pf->sos->successive_approximation;
}

/* Raw image data */
bitstream *get_image_data(parsed_file *pf){
    if (pf->image_data == NULL) {
        fprintf(stderr, "Image data not found\n");
        exit_with_error(pf, NULL);
    }
    return pf->image_data;
}

size_t get_image_data_length(parsed_file *pf){
    if (pf->image_data == NULL) {
        fprintf(stderr, "Image data not found\n");
        exit_with_error(pf, NULL);
    }
    return pf->image_data->taille;
}

/* EOI */
bool get_eoi_marker(parsed_file *pf){
    return pf->eoi_marker;
}

/* Free the parsed file */
void free_parsed_file(parsed_file *pf){
    if (pf == NULL) return;

    for (int i = 0; i < 4; i++) {
        if (pf->tables[i] != NULL) free(pf->tables[i]);
    }

    if (pf->sof != NULL) {
        if (pf->sof->components != NULL) free(pf->sof->components);
        free(pf->sof);
    }

    if (pf->trees != NULL) {
        free_huffman_trees(pf->trees);
    }

    if (pf->sos != NULL) {
        if (pf->sos->components != NULL) free(pf->sos->components);
        free(pf->sos);
    }
    if (pf->image_data != NULL) {
        if (pf->image_data->data != NULL) free(pf->image_data->data);
        free(pf->image_data);
    }
    free(pf);
}
