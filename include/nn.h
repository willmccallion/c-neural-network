#ifndef NN_H
#define NN_H

#include <stdbool.h>

// --- OPTIMIZED ARCHITECTURE (16-32-64) ---
// This is the "Sweet Spot" for CPU training on Merged Datasets.
// It is large enough to learn 70 classes, but small enough to not overfit.

#define OUTPUT_NODES 70

// Block 1: Low-level features (Edges/Lines)
#define CONV1_FILTERS 16
#define CONV1_SIZE 28
#define POOL1_SIZE 14

// Block 2: Mid-level features (Curves/Corners)
#define CONV2_FILTERS 32
#define CONV2_SIZE 14
#define POOL2_SIZE 7

// Block 3: High-level features (Shapes/Objects)
#define CONV3_FILTERS 64
#define CONV3_SIZE 7
#define POOL3_SIZE 3

// Dense Layer: Classification
// 64 filters * 3x3 spatial size = 576 inputs
#define FLATTEN_SIZE (POOL3_SIZE * POOL3_SIZE * CONV3_FILTERS)
#define HIDDEN_NODES 256

typedef struct {
  // Weights
  float *c1_w, *c1_b;
  float *c2_w, *c2_b;
  float *c3_w, *c3_b;
  float *fc1_w, *fc1_b;
  float *fc2_w, *fc2_b;

  // Adam Optimizer State
  float *m_c1_w, *v_c1_w, *m_c1_b, *v_c1_b;
  float *m_c2_w, *v_c2_w, *m_c2_b, *v_c2_b;
  float *m_c3_w, *v_c3_w, *m_c3_b, *v_c3_b;
  float *m_fc1_w, *v_fc1_w, *m_fc1_b, *v_fc1_b;
  float *m_fc2_w, *v_fc2_w, *m_fc2_b, *v_fc2_b;

  long long timestep;

  // Activations (Shared for Visualization)
  float *c1_out;
  float *p1_out;
  float *c2_out;
  float *p2_out;
  float *c3_out;
  float *p3_out;
  float *fc1_out;
  float *final_out;
} NeuralNet;

NeuralNet *nn_create();
void nn_free(NeuralNet *nn);

// Standard forward pass (updates internal viz buffers)
void nn_forward(NeuralNet *nn, float *input_data, bool training);

// Thread-safe inference (allocates temp buffers, returns probabilities)
void nn_inference(NeuralNet *nn, float *input, float *output_probs);

float nn_train_batch(NeuralNet *nn, float *batch_input, float *batch_target,
                     int batch_size, float lr);

void nn_save(NeuralNet *nn, const char *filename);
NeuralNet *nn_load(const char *filename);
NeuralNet *nn_clone(NeuralNet *src);

#endif
