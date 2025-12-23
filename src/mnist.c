#include "mnist.h"
#include <stdio.h>
#include <stdlib.h>

unsigned int swap_endian(unsigned int i) {
  return ((i >> 24) & 0xff) | ((i >> 8) & 0xff00) | ((i << 8) & 0xff0000) |
         ((i << 24) & 0xff000000);
}

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

  // Images
  fread(&magic, 4, 1, f_img);
  fread(&count, 4, 1, f_img);
  fread(&rows, 4, 1, f_img);
  fread(&cols, 4, 1, f_img);

  data->count = swap_endian(count);
  data->width = swap_endian(rows);
  data->height = swap_endian(cols);

  // Labels
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

void free_mnist(MnistData *data) {
  if (!data)
    return;
  for (int i = 0; i < data->count; i++)
    free(data->images[i]);
  free(data->images);
  free(data->labels);
  free(data);
}
