#ifndef MNIST_H
#define MNIST_H

typedef struct {
  int count;
  int width;
  int height;
  unsigned char **images;
  unsigned char *labels;
} MnistData;

MnistData *load_mnist(const char *image_path, const char *label_path);
void free_mnist(MnistData *data);

#endif
