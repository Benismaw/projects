#ifndef __LECTURE_ENTETE_H__
#define __LECTURE_ENTETE_H__

#include <stdint.h>
#include <stdbool.h>
#include "arbre.h"
#include "bitstream.h"


typedef struct parsed_file parsed_file;

// Parses a JPEG file
parsed_file *parse_file(char *name);

void parse_markers(parsed_file *pf, FILE *f, uint8_t marker[2]);

// Frees the memory allocated for the parsed file.
void free_parsed_file(parsed_file *pf);

// Structure representing a Quantization Table (DQT segment in JPEG).
typedef struct {
    uint8_t precision;  // 0 = 8-bit, 1 = 16-bit
    uint8_t iq;         // Index of the table (0-3)
    uint8_t table[64]; // Quantization values
} QuantizationTable;

// Structure representing a Huffman Table (DHT segment in JPEG).

typedef struct HuffmanTable HuffmanTable;

typedef struct Node Node;

typedef struct huffman_trees huffman_trees;

// Structure representing the bitstream of the raw image data.
typedef struct bitstream bitstream;

/* SOI */
// Returns true if the SOI marker is found, false otherwise.
bool get_soi_marker(parsed_file *pf);


/* DQT */
// Retrieves a Quantization Table (DQT) from the parsed file, or NULL if not found.
QuantizationTable *get_dqt(parsed_file *pf, uint8_t index);


/* SOF */
// Get the precision of the image (always 8 bits in baseline JPEG)
uint8_t get_precision(parsed_file *pf);

// Get the width and height of the image
void get_image_size(parsed_file *pf, uint16_t *width, uint16_t *height);

// Get the number of components in the image (components Y, Cb, Cr)
uint8_t get_nb_components_sof(parsed_file *pf);

// Get the component ic of a specific component 
// (0 = Y, 1 = Cb, 2 = Cr) (0 only in grayscale)
uint8_t get_component_ic_sof(parsed_file *pf, uint8_t index);

// Get the horizontal and verticall sampling factor of a specific component
void get_sampling_factor_sof(parsed_file *pf, uint8_t index, uint8_t *horizontal, uint8_t *vertical);

// Get the quantization table IQ of a specific component
uint8_t get_quantization_table_iq_sof(parsed_file *pf, uint8_t index);


/* DHT */
// Retrieves the huffman trees (DC and AC) from the parsed file.
huffman_trees *get_dht(parsed_file *pf);


/* SOS */
// Get the number of components in the image (components Y, Cb, Cr)
uint8_t get_nb_components_sos(parsed_file *pf);

// Get the component ic of a specific component
// (0 = Y, 1 = Cb, 2 = Cr) (0 only in grayscale)
uint8_t get_component_ic_sos(parsed_file *pf, uint8_t index);

// Get the DC and AC Huffman table ID of a specific component
// (0 = Y, 1 = Cb, 2 = Cr) (0 only in grayscale)
void get_huffman_table_id_sos(parsed_file *pf, uint8_t index, uint8_t *dc, uint8_t *ac);

// Get the start and end of the selection (0 and 63 in baseline JPEG)
void get_start_end_selection_sos(parsed_file *pf, uint8_t *start, uint8_t *end);

// Get the successive approximation (0 in baseline JPEG)
uint8_t get_successive_approximation_sos(parsed_file *pf);


/* Raw image data */
// Get raw image data + length + position = 0
bitstream *get_image_data(parsed_file *pf);

// Get the length of the raw image data
size_t get_image_data_length(parsed_file *pf);


/* EOI */
// Returns true if the EOI marker is found, false otherwise.
bool get_eoi_marker(parsed_file *pf);

#endif
