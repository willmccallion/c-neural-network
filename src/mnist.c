/**
 * @file mnist.c
 * @brief MNIST Dataset Loading and Management
 *
 * This module handles loading and parsing of MNIST-format dataset files, which
 * store image and label data in a binary format with big-endian integer
 * encoding. It provides functions to read the dataset from disk, manage memory
 * for image arrays, and properly deallocate resources when the dataset is no
 * longer needed.
 */

#include "mnist.h"
#include <stdio.h>
#include <stdlib.h>

/**
 * Swaps the byte order of a 32-bit unsigned integer from big-endian to
 * little-endian.
 *
 * MNIST dataset files store multi-byte integers in big-endian format, which
 * must be converted to the host machine's native byte order for correct
 * interpretation. This function performs a full 32-bit byte swap, reversing the
 * order of all four bytes in the integer value.
 *
 * @param i The big-endian integer value to convert
 * @return The same integer value in little-endian byte order
 */
unsigned int swap_endian(unsigned int i) {
  return ((i >> 24) & 0xff) | ((i >> 8) & 0xff00) | ((i << 8) & 0xff0000) |
         ((i << 24) & 0xff000000);
}

/**
 * Loads a MNIST dataset from image and label files.
 *
 * Opens and parses the MNIST binary format files, reading the magic numbers,
 * image count, dimensions, and pixel/label data. Allocates memory for the
 * MnistData structure and individual image arrays. The function handles
 * endianness conversion for multi-byte integers and validates file access.
 * Returns NULL if either file cannot be opened, allowing the caller to handle
 * missing dataset files gracefully.
 *
 * @param image_path Path to the MNIST image file (typically
 * *-images-idx3-ubyte)
 * @param label_path Path to the MNIST label file (typically
 * *-labels-idx1-ubyte)
 * @return Pointer to allocated MnistData structure, or NULL on failure
 */
MnistData *load_mnist(const char *image_path, const char *label_path) {
  FILE *f_img = fopen(image_path, "rb");
  FILE *f_lbl = fopen(label_path, "rb");

  if (!f_img || !f_lbl) {
    if (f_img)
      fclose(f_img);
    if (f_lbl)
      fclose(f_lbl);
    return NULL;
  }

  MnistData *data = malloc(sizeof(MnistData));
  unsigned int magic, count, rows, cols;

  fread(&magic, 4, 1, f_img);
  fread(&count, 4, 1, f_img);
  fread(&rows, 4, 1, f_img);
  fread(&cols, 4, 1, f_img);

  data->count = swap_endian(count);
  data->width = swap_endian(rows);
  data->height = swap_endian(cols);

  unsigned int l_magic, l_count;
  fread(&l_magic, 4, 1, f_lbl);
  fread(&l_count, 4, 1, f_lbl);

  data->images = malloc(data->count * sizeof(unsigned char *));
  data->labels = malloc(data->count * sizeof(unsigned char));

  int image_size = data->width * data->height;
  for (int i = 0; i < data->count; i++) {
    data->images[i] = malloc(image_size);
    fread(data->images[i], 1, image_size, f_img);
    fread(&data->labels[i], 1, 1, f_lbl);
  }

  fclose(f_img);
  fclose(f_lbl);
  return data;
}

/**
 * Deallocates all memory associated with a MnistData structure.
 *
 * Frees the individual image arrays, the image pointer array, the label array,
 * and finally the MnistData structure itself. This function safely handles NULL
 * pointers and ensures complete cleanup of all dynamically allocated resources
 * to prevent memory leaks. Must be called for every dataset loaded with
 * load_mnist before the dataset pointer goes out of scope.
 *
 * @param data Pointer to the MnistData structure to deallocate, or NULL (no-op)
 */
void free_mnist(MnistData *data) {
  if (!data)
    return;
  for (int i = 0; i < data->count; i++)
    free(data->images[i]);
  free(data->images);
  free(data->labels);
  free(data);
}
