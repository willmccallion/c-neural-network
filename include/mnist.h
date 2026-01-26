/**
 * @file mnist.h
 * @brief MNIST Dataset Structure and Loading Functions
 *
 * This header defines the data structure for holding MNIST-format dataset
 * information and declares functions for loading dataset files from disk and
 * managing associated memory resources.
 */

#ifndef MNIST_H
#define MNIST_H

/**
 * Structure holding a loaded MNIST dataset in memory.
 *
 * This structure contains all image and label data from a MNIST-format dataset
 * file. Images are stored as arrays of unsigned bytes (0-255 pixel values),
 * with each image allocated as a separate array. Labels are stored as a single
 * array of unsigned bytes representing class indices. The structure must be
 * freed using free_mnist() to prevent memory leaks.
 */
typedef struct {
  /**
   * Number of images in the dataset.
   */
  int count;

  /**
   * Width of each image in pixels (typically 28 for MNIST).
   */
  int width;

  /**
   * Height of each image in pixels (typically 28 for MNIST).
   */
  int height;

  /**
   * Array of pointers to individual image arrays. Each image is stored as a
   * contiguous array of width * height unsigned bytes.
   */
  unsigned char **images;

  /**
   * Array of label values, one per image. Each label is an unsigned byte
   * representing the class index (0-9 for standard MNIST digits).
   */
  unsigned char *labels;
} MnistData;

/**
 * Loads a MNIST dataset from image and label files.
 *
 * @param image_path Path to the MNIST image file (typically
 * *-images-idx3-ubyte)
 * @param label_path Path to the MNIST label file (typically
 * *-labels-idx1-ubyte)
 * @return Pointer to allocated MnistData structure, or NULL on failure
 */
MnistData *load_mnist(const char *image_path, const char *label_path);

/**
 * Deallocates all memory associated with a MnistData structure.
 *
 * @param data Pointer to the MnistData structure to deallocate, or NULL (no-op)
 */
void free_mnist(MnistData *data);

#endif
