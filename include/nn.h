/**
 * @file nn.h
 * @brief Neural Network Architecture Definition and Interface
 *
 * This header defines the neural network architecture constants, the NeuralNet
 * structure containing all network parameters and state, and the function
 * interface for network creation, training, inference, and persistence. The
 * architecture consists of three convolutional layers with max-pooling followed
 * by two fully-connected layers, optimized for CPU-based training on 70-class
 * classification tasks.
 */

#ifndef NN_H
#define NN_H

#include <stdbool.h>

/**
 * Number of output classes in the classification task.
 *
 * Supports 70 classes: digits (0-9), uppercase letters (A-Z), lowercase letters
 * (a, b, d, e, f, g, h, n, q, r, t), and drawing categories (apple, book,
 * etc.).
 */
#define OUTPUT_NODES 70

/**
 * Number of filters in the first convolutional layer.
 *
 * This layer extracts low-level features such as edges and lines from the
 * 28x28 input image.
 */
#define CONV1_FILTERS 16

/**
 * Spatial dimension of the first convolutional layer output (28x28).
 *
 * Maintains the same size as the input due to same-padding in the convolution.
 */
#define CONV1_SIZE 28

/**
 * Spatial dimension after the first max-pooling operation (14x14).
 *
 * Reduces the 28x28 feature maps to 14x14 through 2x2 max-pooling.
 */
#define POOL1_SIZE 14

/**
 * Number of filters in the second convolutional layer.
 *
 * This layer processes mid-level features such as curves and corners from
 * the 14x14 pooled feature maps.
 */
#define CONV2_FILTERS 32

/**
 * Spatial dimension of the second convolutional layer output (14x14).
 *
 * Maintains the same size as the input due to same-padding in the convolution.
 */
#define CONV2_SIZE 14

/**
 * Spatial dimension after the second max-pooling operation (7x7).
 *
 * Reduces the 14x14 feature maps to 7x7 through 2x2 max-pooling.
 */
#define POOL2_SIZE 7

/**
 * Number of filters in the third convolutional layer.
 *
 * This layer extracts high-level features such as shapes and object parts
 * from the 7x7 pooled feature maps.
 */
#define CONV3_FILTERS 64

/**
 * Spatial dimension of the third convolutional layer output (7x7).
 *
 * Maintains the same size as the input due to same-padding in the convolution.
 */
#define CONV3_SIZE 7

/**
 * Spatial dimension after the third max-pooling operation (3x3).
 *
 * Reduces the 7x7 feature maps to 3x3 through 2x2 max-pooling, with the
 * final spatial dimension being 3 due to integer division.
 */
#define POOL3_SIZE 3

/**
 * Number of elements in the flattened feature vector fed to the dense layer.
 *
 * Calculated as POOL3_SIZE * POOL3_SIZE * CONV3_FILTERS = 3 * 3 * 64 = 576.
 * This represents the total number of activations from the final pooling layer.
 */
#define FLATTEN_SIZE (POOL3_SIZE * POOL3_SIZE * CONV3_FILTERS)

/**
 * Number of neurons in the first fully-connected (hidden) layer.
 *
 * This dense layer processes the flattened convolutional features and produces
 * a 256-dimensional representation for final classification.
 */
#define HIDDEN_NODES 256

/**
 * Neural network structure containing all parameters, optimizer state, and
 * activations.
 *
 * This structure holds all trainable parameters (weights and biases) for the
 * convolutional and fully-connected layers, Adam optimizer state (momentum and
 * velocity buffers) for adaptive learning rate updates, and activation buffers
 * used for visualization and forward propagation. The network architecture
 * follows a hierarchical feature extraction pattern: three convolutional blocks
 * extract increasingly abstract features, which are then classified by two
 * dense layers.
 */
typedef struct {
  /**
   * Weight matrix for the first convolutional layer.
   *
   * Stored as a flat array of size CONV1_FILTERS * 9, where each filter has
   * 9 weights (3x3 kernel) for a single input channel.
   */
  float *c1_w;

  /**
   * Bias vector for the first convolutional layer.
   *
   * Array of CONV1_FILTERS elements, one bias per output filter.
   */
  float *c1_b;

  /**
   * Weight matrix for the second convolutional layer.
   *
   * Stored as a flat array of size CONV2_FILTERS * CONV1_FILTERS * 9, where
   * each of CONV2_FILTERS filters has CONV1_FILTERS * 9 weights (one 3x3
   * kernel per input feature map).
   */
  float *c2_w;

  /**
   * Bias vector for the second convolutional layer.
   *
   * Array of CONV2_FILTERS elements, one bias per output filter.
   */
  float *c2_b;

  /**
   * Weight matrix for the third convolutional layer.
   *
   * Stored as a flat array of size CONV3_FILTERS * CONV2_FILTERS * 9, where
   * each of CONV3_FILTERS filters has CONV2_FILTERS * 9 weights (one 3x3
   * kernel per input feature map).
   */
  float *c3_w;

  /**
   * Bias vector for the third convolutional layer.
   *
   * Array of CONV3_FILTERS elements, one bias per output filter.
   */
  float *c3_b;

  /**
   * Weight matrix for the first fully-connected layer.
   *
   * Stored as a flat array of size FLATTEN_SIZE * HIDDEN_NODES, organized
   * in row-major order where each row corresponds to one input element and
   * each column corresponds to one hidden neuron.
   */
  float *fc1_w;

  /**
   * Bias vector for the first fully-connected layer.
   *
   * Array of HIDDEN_NODES elements, one bias per hidden neuron.
   */
  float *fc1_b;

  /**
   * Weight matrix for the second fully-connected (output) layer.
   *
   * Stored as a flat array of size HIDDEN_NODES * OUTPUT_NODES, organized
   * in row-major order where each row corresponds to one hidden neuron and
   * each column corresponds to one output class.
   */
  float *fc2_w;

  /**
   * Bias vector for the second fully-connected (output) layer.
   *
   * Array of OUTPUT_NODES elements, one bias per output class.
   */
  float *fc2_b;

  /**
   * Momentum buffer for first convolutional layer weights (Adam optimizer).
   *
   * Maintains exponential moving average of gradients for adaptive learning
   * rate computation. Same size as c1_w.
   */
  float *m_c1_w;

  /**
   * Velocity buffer for first convolutional layer weights (Adam optimizer).
   *
   * Maintains exponential moving average of squared gradients for adaptive
   * learning rate computation. Same size as c1_w.
   */
  float *v_c1_w;

  /**
   * Momentum buffer for first convolutional layer biases (Adam optimizer).
   *
   * Same size as c1_b.
   */
  float *m_c1_b;

  /**
   * Velocity buffer for first convolutional layer biases (Adam optimizer).
   *
   * Same size as c1_b.
   */
  float *v_c1_b;

  /**
   * Momentum buffer for second convolutional layer weights (Adam optimizer).
   *
   * Same size as c2_w.
   */
  float *m_c2_w;

  /**
   * Velocity buffer for second convolutional layer weights (Adam optimizer).
   *
   * Same size as c2_w.
   */
  float *v_c2_w;

  /**
   * Momentum buffer for second convolutional layer biases (Adam optimizer).
   *
   * Same size as c2_b.
   */
  float *m_c2_b;

  /**
   * Velocity buffer for second convolutional layer biases (Adam optimizer).
   *
   * Same size as c2_b.
   */
  float *v_c2_b;

  /**
   * Momentum buffer for third convolutional layer weights (Adam optimizer).
   *
   * Same size as c3_w.
   */
  float *m_c3_w;

  /**
   * Velocity buffer for third convolutional layer weights (Adam optimizer).
   *
   * Same size as c3_w.
   */
  float *v_c3_w;

  /**
   * Momentum buffer for third convolutional layer biases (Adam optimizer).
   *
   * Same size as c3_b.
   */
  float *m_c3_b;

  /**
   * Velocity buffer for third convolutional layer biases (Adam optimizer).
   *
   * Same size as c3_b.
   */
  float *v_c3_b;

  /**
   * Momentum buffer for first fully-connected layer weights (Adam optimizer).
   *
   * Same size as fc1_w.
   */
  float *m_fc1_w;

  /**
   * Velocity buffer for first fully-connected layer weights (Adam optimizer).
   *
   * Same size as fc1_w.
   */
  float *v_fc1_w;

  /**
   * Momentum buffer for first fully-connected layer biases (Adam optimizer).
   *
   * Same size as fc1_b.
   */
  float *m_fc1_b;

  /**
   * Velocity buffer for first fully-connected layer biases (Adam optimizer).
   *
   * Same size as fc1_b.
   */
  float *v_fc1_b;

  /**
   * Momentum buffer for second fully-connected layer weights (Adam optimizer).
   *
   * Same size as fc2_w.
   */
  float *m_fc2_w;

  /**
   * Velocity buffer for second fully-connected layer weights (Adam optimizer).
   *
   * Same size as fc2_w.
   */
  float *v_fc2_w;

  /**
   * Momentum buffer for second fully-connected layer biases (Adam optimizer).
   *
   * Same size as fc2_b.
   */
  float *m_fc2_b;

  /**
   * Velocity buffer for second fully-connected layer biases (Adam optimizer).
   *
   * Same size as fc2_b.
   */
  float *v_fc2_b;

  /**
   * Current timestep counter for Adam optimizer bias correction.
   *
   * Incremented after each batch training step, used to compute bias-corrected
   * momentum and velocity estimates that account for the initial zero-state
   * of the exponential moving averages.
   */
  long long timestep;

  /**
   * Output activations from the first convolutional layer
   * (28x28xCONV1_FILTERS).
   *
   * Used for visualization of low-level feature detection. Updated during
   * forward propagation when using nn_forward().
   */
  float *c1_out;

  /**
   * Output activations from the first max-pooling layer (14x14xCONV1_FILTERS).
   *
   * Used for visualization and as input to the second convolutional layer.
   */
  float *p1_out;

  /**
   * Output activations from the second convolutional layer
   * (14x14xCONV2_FILTERS).
   *
   * Used for visualization of mid-level feature detection.
   */
  float *c2_out;

  /**
   * Output activations from the second max-pooling layer (7x7xCONV2_FILTERS).
   *
   * Used for visualization and as input to the third convolutional layer.
   */
  float *p2_out;

  /**
   * Output activations from the third convolutional layer (7x7xCONV3_FILTERS).
   *
   * Used for visualization of high-level feature detection.
   */
  float *c3_out;

  /**
   * Output activations from the third max-pooling layer (3x3xCONV3_FILTERS).
   *
   * Flattened and used as input to the first fully-connected layer.
   */
  float *p3_out;

  /**
   * Output activations from the first fully-connected (hidden) layer.
   *
   * Array of HIDDEN_NODES elements, used for visualization of dense layer
   * activations and as input to the output layer.
   */
  float *fc1_out;

  /**
   * Final output probabilities after softmax normalization.
   *
   * Array of OUTPUT_NODES elements representing the probability distribution
   * over all classes. Values sum to 1.0 and represent the network's confidence
   * in each class prediction.
   */
  float *final_out;
} NeuralNet;

/**
 * Creates and initializes a new neural network instance.
 *
 * @return Pointer to the newly allocated and initialized NeuralNet structure
 */
NeuralNet *nn_create();

/**
 * Deallocates all memory associated with a neural network instance.
 *
 * @param nn Pointer to the NeuralNet structure to deallocate, or NULL (no-op)
 */
void nn_free(NeuralNet *nn);

/**
 * Performs forward propagation using the network's internal activation buffers.
 *
 * @param nn Pointer to the neural network structure
 * @param input_data Pointer to the 28x28 input image (784 elements, normalized
 * 0-1)
 * @param training Boolean flag indicating training mode (currently unused)
 */
void nn_forward(NeuralNet *nn, float *input_data, bool training);

/**
 * Performs thread-safe inference using temporary activation buffers.
 *
 * @param nn Pointer to the neural network structure (weights are read-only)
 * @param input Pointer to the 28x28 input image (784 elements, normalized 0-1)
 * @param output_probs Pointer to the output buffer for class probabilities
 * (OUTPUT_NODES elements)
 */
void nn_inference(NeuralNet *nn, float *input, float *output_probs);

/**
 * Trains the network on a batch of examples using backpropagation and Adam
 * optimization.
 *
 * @param nn Pointer to the neural network structure (weights are modified)
 * @param batch_input Pointer to batch input images (batch_size x 784 elements)
 * @param batch_target Pointer to batch target labels as one-hot vectors
 * (batch_size x OUTPUT_NODES)
 * @param batch_size Number of examples in the batch
 * @param lr Learning rate for Adam optimizer updates
 * @return Average cross-entropy loss over the batch
 */
float nn_train_batch(NeuralNet *nn, float *batch_input, float *batch_target,
                     int batch_size, float lr);

/**
 * Saves the network weights and biases to a binary file.
 *
 * @param nn Pointer to the neural network structure to save
 * @param filename Path to the output file (will be overwritten if it exists)
 */
void nn_save(NeuralNet *nn, const char *filename);

/**
 * Loads network weights and biases from a binary file.
 *
 * @param filename Path to the model file to load
 * @return Pointer to the loaded NeuralNet structure, or NULL on failure
 */
NeuralNet *nn_load(const char *filename);

/**
 * Creates a deep copy of a neural network, duplicating all weights and biases.
 *
 * @param src Pointer to the source neural network to clone
 * @return Pointer to the newly allocated cloned network
 */
NeuralNet *nn_clone(NeuralNet *src);

#endif
