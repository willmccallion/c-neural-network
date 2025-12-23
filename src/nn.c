#include "nn.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Math helpers
float get_random_weight(float n_inputs) {
  float std_dev = sqrtf(2.0f / n_inputs);
  float r = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
  return r * std_dev;
}

float relu(float x) { return x > 0 ? x : x * 0.01f; } // Leaky ReLU
float d_relu(float x) { return x > 0 ? 1.0f : 0.01f; }

void softmax(float *input, int n) {
  float max = input[0];
  for (int i = 1; i < n; i++)
    if (input[i] > max)
      max = input[i];
  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    input[i] = expf(input[i] - max);
    sum += input[i];
  }
  for (int i = 0; i < n; i++)
    input[i] /= (sum + 1e-9f);
}

void alloc_tensor(float **ptr, int size) { *ptr = calloc(size, sizeof(float)); }

NeuralNet *nn_create() {
  NeuralNet *nn = (NeuralNet *)malloc(sizeof(NeuralNet));
  nn->timestep = 0;

  // Sizes
  int s_c1_w = CONV1_FILTERS * 9;
  int s_c2_w = CONV2_FILTERS * CONV1_FILTERS * 9;
  int s_c3_w = CONV3_FILTERS * CONV2_FILTERS * 9;
  int s_fc1_w = FLATTEN_SIZE * HIDDEN_NODES;
  int s_fc2_w = HIDDEN_NODES * OUTPUT_NODES;

  // Weights
  nn->c1_w = malloc(s_c1_w * sizeof(float));
  nn->c1_b = calloc(CONV1_FILTERS, sizeof(float));
  nn->c2_w = malloc(s_c2_w * sizeof(float));
  nn->c2_b = calloc(CONV2_FILTERS, sizeof(float));
  nn->c3_w = malloc(s_c3_w * sizeof(float));
  nn->c3_b = calloc(CONV3_FILTERS, sizeof(float));
  nn->fc1_w = malloc(s_fc1_w * sizeof(float));
  nn->fc1_b = calloc(HIDDEN_NODES, sizeof(float));
  nn->fc2_w = malloc(s_fc2_w * sizeof(float));
  nn->fc2_b = calloc(OUTPUT_NODES, sizeof(float));

  // Init weights
  for (int i = 0; i < s_c1_w; i++)
    nn->c1_w[i] = get_random_weight(9.0f);
  for (int i = 0; i < s_c2_w; i++)
    nn->c2_w[i] = get_random_weight(CONV1_FILTERS * 9.0f);
  for (int i = 0; i < s_c3_w; i++)
    nn->c3_w[i] = get_random_weight(CONV2_FILTERS * 9.0f);
  for (int i = 0; i < s_fc1_w; i++)
    nn->fc1_w[i] = get_random_weight(FLATTEN_SIZE);
  for (int i = 0; i < s_fc2_w; i++)
    nn->fc2_w[i] = get_random_weight(HIDDEN_NODES);

  // Adam optimizer state
  alloc_tensor(&nn->m_c1_w, s_c1_w);
  alloc_tensor(&nn->v_c1_w, s_c1_w);
  alloc_tensor(&nn->m_c1_b, CONV1_FILTERS);
  alloc_tensor(&nn->v_c1_b, CONV1_FILTERS);
  alloc_tensor(&nn->m_c2_w, s_c2_w);
  alloc_tensor(&nn->v_c2_w, s_c2_w);
  alloc_tensor(&nn->m_c2_b, CONV2_FILTERS);
  alloc_tensor(&nn->v_c2_b, CONV2_FILTERS);
  alloc_tensor(&nn->m_c3_w, s_c3_w);
  alloc_tensor(&nn->v_c3_w, s_c3_w);
  alloc_tensor(&nn->m_c3_b, CONV3_FILTERS);
  alloc_tensor(&nn->v_c3_b, CONV3_FILTERS);
  alloc_tensor(&nn->m_fc1_w, s_fc1_w);
  alloc_tensor(&nn->v_fc1_w, s_fc1_w);
  alloc_tensor(&nn->m_fc1_b, HIDDEN_NODES);
  alloc_tensor(&nn->v_fc1_b, HIDDEN_NODES);
  alloc_tensor(&nn->m_fc2_w, s_fc2_w);
  alloc_tensor(&nn->v_fc2_w, s_fc2_w);
  alloc_tensor(&nn->m_fc2_b, OUTPUT_NODES);
  alloc_tensor(&nn->v_fc2_b, OUTPUT_NODES);

  // Activations (for visualization)
  alloc_tensor(&nn->c1_out, CONV1_SIZE * CONV1_SIZE * CONV1_FILTERS);
  alloc_tensor(&nn->p1_out, POOL1_SIZE * POOL1_SIZE * CONV1_FILTERS);
  alloc_tensor(&nn->c2_out, CONV2_SIZE * CONV2_SIZE * CONV2_FILTERS);
  alloc_tensor(&nn->p2_out, POOL2_SIZE * POOL2_SIZE * CONV2_FILTERS);
  alloc_tensor(&nn->c3_out, CONV3_SIZE * CONV3_SIZE * CONV3_FILTERS);
  alloc_tensor(&nn->p3_out, POOL3_SIZE * POOL3_SIZE * CONV3_FILTERS);
  alloc_tensor(&nn->fc1_out, HIDDEN_NODES);
  alloc_tensor(&nn->final_out, OUTPUT_NODES);

  return nn;
}

void nn_free(NeuralNet *nn) {
  if (!nn)
    return;
  free(nn->c1_w);
  free(nn->c1_b);
  free(nn->c2_w);
  free(nn->c2_b);
  free(nn->c3_w);
  free(nn->c3_b);
  free(nn->fc1_w);
  free(nn->fc1_b);
  free(nn->fc2_w);
  free(nn->fc2_b);

  free(nn->m_c1_w);
  free(nn->v_c1_w);
  free(nn->m_c1_b);
  free(nn->v_c1_b);
  free(nn->m_c2_w);
  free(nn->v_c2_w);
  free(nn->m_c2_b);
  free(nn->v_c2_b);
  free(nn->m_c3_w);
  free(nn->v_c3_w);
  free(nn->m_c3_b);
  free(nn->v_c3_b);
  free(nn->m_fc1_w);
  free(nn->v_fc1_w);
  free(nn->m_fc1_b);
  free(nn->v_fc1_b);
  free(nn->m_fc2_w);
  free(nn->v_fc2_w);
  free(nn->m_fc2_b);
  free(nn->v_fc2_b);

  free(nn->c1_out);
  free(nn->p1_out);
  free(nn->c2_out);
  free(nn->p2_out);
  free(nn->c3_out);
  free(nn->p3_out);
  free(nn->fc1_out);
  free(nn->final_out);
  free(nn);
}

// Forward Operations
void conv3x3(float *in, float *w, float *b, float *out, int d, int in_f,
             int out_f) {
  for (int f = 0; f < out_f; f++) {
    for (int y = 0; y < d; y++) {
      for (int x = 0; x < d; x++) {
        float sum = b[f];
        for (int i = 0; i < in_f; i++) {
          int in_off = i * (d * d);
          int w_off = f * (in_f * 9) + i * 9;
          for (int ky = 0; ky < 3; ky++) {
            int iy = y + ky - 1;
            if (iy >= 0 && iy < d) {
              for (int kx = 0; kx < 3; kx++) {
                int ix = x + kx - 1;
                if (ix >= 0 && ix < d) {
                  sum += in[in_off + iy * d + ix] * w[w_off + ky * 3 + kx];
                }
              }
            }
          }
        }
        out[f * (d * d) + y * d + x] = relu(sum);
      }
    }
  }
}

void maxpool(float *in, float *out, int in_d, int f) {
  int out_d = in_d / 2;
  for (int i = 0; i < f; i++) {
    for (int y = 0; y < out_d; y++) {
      for (int x = 0; x < out_d; x++) {
        float max = -1e9f;
        int off = i * (in_d * in_d);
        for (int ky = 0; ky < 2; ky++) {
          for (int kx = 0; kx < 2; kx++) {
            float v = in[off + (y * 2 + ky) * in_d + (x * 2 + kx)];
            if (v > max)
              max = v;
          }
        }
        out[i * (out_d * out_d) + y * out_d + x] = max;
      }
    }
  }
}

void nn_forward_impl(NeuralNet *nn, float *input, float *c1, float *p1,
                     float *c2, float *p2, float *c3, float *p3, float *fc1,
                     float *final) {
  conv3x3(input, nn->c1_w, nn->c1_b, c1, 28, 1, CONV1_FILTERS);
  maxpool(c1, p1, 28, CONV1_FILTERS);

  conv3x3(p1, nn->c2_w, nn->c2_b, c2, 14, CONV1_FILTERS, CONV2_FILTERS);
  maxpool(c2, p2, 14, CONV2_FILTERS);

  conv3x3(p2, nn->c3_w, nn->c3_b, c3, 7, CONV2_FILTERS, CONV3_FILTERS);
  maxpool(c3, p3, 7, CONV3_FILTERS);

  for (int h = 0; h < HIDDEN_NODES; h++) {
    float sum = nn->fc1_b[h];
    for (int i = 0; i < FLATTEN_SIZE; i++)
      sum += p3[i] * nn->fc1_w[i * HIDDEN_NODES + h];
    fc1[h] = relu(sum);
  }

  for (int o = 0; o < OUTPUT_NODES; o++) {
    float sum = nn->fc2_b[o];
    for (int h = 0; h < HIDDEN_NODES; h++)
      sum += fc1[h] * nn->fc2_w[h * OUTPUT_NODES + o];
    final[o] = sum;
  }
  softmax(final, OUTPUT_NODES);
}

void nn_forward(NeuralNet *nn, float *input_data, bool training) {
  nn_forward_impl(nn, input_data, nn->c1_out, nn->p1_out, nn->c2_out,
                  nn->p2_out, nn->c3_out, nn->p3_out, nn->fc1_out,
                  nn->final_out);
}

void nn_inference(NeuralNet *nn, float *input, float *output_probs) {
  float *t_c1 = malloc(CONV1_SIZE * CONV1_SIZE * CONV1_FILTERS * sizeof(float));
  float *t_p1 = malloc(POOL1_SIZE * POOL1_SIZE * CONV1_FILTERS * sizeof(float));
  float *t_c2 = malloc(CONV2_SIZE * CONV2_SIZE * CONV2_FILTERS * sizeof(float));
  float *t_p2 = malloc(POOL2_SIZE * POOL2_SIZE * CONV2_FILTERS * sizeof(float));
  float *t_c3 = malloc(CONV3_SIZE * CONV3_SIZE * CONV3_FILTERS * sizeof(float));
  float *t_p3 = malloc(POOL3_SIZE * POOL3_SIZE * CONV3_FILTERS * sizeof(float));
  float *t_fc1 = malloc(HIDDEN_NODES * sizeof(float));
  float *t_out = malloc(OUTPUT_NODES * sizeof(float));

  nn_forward_impl(nn, input, t_c1, t_p1, t_c2, t_p2, t_c3, t_p3, t_fc1, t_out);
  memcpy(output_probs, t_out, OUTPUT_NODES * sizeof(float));

  free(t_c1);
  free(t_p1);
  free(t_c2);
  free(t_p2);
  free(t_c3);
  free(t_p3);
  free(t_fc1);
  free(t_out);
}

// Backpropagation helpers
void backprop_pool(float *grad_out, float *grad_in, float *input_data, int in_d,
                   int f) {
  int out_d = in_d / 2;
  for (int i = 0; i < f; i++) {
    for (int y = 0; y < out_d; y++) {
      for (int x = 0; x < out_d; x++) {
        float g = grad_out[i * (out_d * out_d) + y * out_d + x];
        if (g == 0)
          continue;

        int off = i * (in_d * in_d);
        float max_val = -1e9f;
        int max_idx = 0;

        for (int ky = 0; ky < 2; ky++) {
          for (int kx = 0; kx < 2; kx++) {
            int idx = off + (y * 2 + ky) * in_d + (x * 2 + kx);
            if (input_data[idx] > max_val) {
              max_val = input_data[idx];
              max_idx = idx;
            }
          }
        }
        grad_in[max_idx] += g;
      }
    }
  }
}

void adam(float *w, float *m, float *v, float *g, int n, float lr,
          long long t) {
  float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
  for (int i = 0; i < n; i++) {
    m[i] = b1 * m[i] + (1 - b1) * g[i];
    v[i] = b2 * v[i] + (1 - b2) * (g[i] * g[i]);
    float mh = m[i] / (1 - powf(b1, t));
    float vh = v[i] / (1 - powf(b2, t));
    w[i] -= lr * mh / (sqrtf(vh) + eps);
  }
}

// Training loop
float nn_train_batch(NeuralNet *nn, float *batch_input, float *batch_target,
                     int batch_size, float lr) {
  nn->timestep++;
  float total_loss = 0.0f;

  // Gradient Accumulators
  int sz_c1w = CONV1_FILTERS * 9;
  int sz_c2w = CONV2_FILTERS * CONV1_FILTERS * 9;
  int sz_c3w = CONV3_FILTERS * CONV2_FILTERS * 9;
  int sz_fc1w = FLATTEN_SIZE * HIDDEN_NODES;
  int sz_fc2w = HIDDEN_NODES * OUTPUT_NODES;

  float *g_c1w = calloc(sz_c1w, sizeof(float));
  float *g_c1b = calloc(CONV1_FILTERS, sizeof(float));
  float *g_c2w = calloc(sz_c2w, sizeof(float));
  float *g_c2b = calloc(CONV2_FILTERS, sizeof(float));
  float *g_c3w = calloc(sz_c3w, sizeof(float));
  float *g_c3b = calloc(CONV3_FILTERS, sizeof(float));
  float *g_fc1w = calloc(sz_fc1w, sizeof(float));
  float *g_fc1b = calloc(HIDDEN_NODES, sizeof(float));
  float *g_fc2w = calloc(sz_fc2w, sizeof(float));
  float *g_fc2b = calloc(OUTPUT_NODES, sizeof(float));

#pragma omp parallel
  {
    // Thread-local buffers
    float *t_c1 =
        malloc(CONV1_SIZE * CONV1_SIZE * CONV1_FILTERS * sizeof(float));
    float *t_p1 =
        malloc(POOL1_SIZE * POOL1_SIZE * CONV1_FILTERS * sizeof(float));
    float *t_c2 =
        malloc(CONV2_SIZE * CONV2_SIZE * CONV2_FILTERS * sizeof(float));
    float *t_p2 =
        malloc(POOL2_SIZE * POOL2_SIZE * CONV2_FILTERS * sizeof(float));
    float *t_c3 =
        malloc(CONV3_SIZE * CONV3_SIZE * CONV3_FILTERS * sizeof(float));
    float *t_p3 =
        malloc(POOL3_SIZE * POOL3_SIZE * CONV3_FILTERS * sizeof(float));
    float *t_fc1 = malloc(HIDDEN_NODES * sizeof(float));
    float *t_out = malloc(OUTPUT_NODES * sizeof(float));

    // Thread-local gradients
    float *l_c1w = calloc(sz_c1w, sizeof(float));
    float *l_c1b = calloc(CONV1_FILTERS, sizeof(float));
    float *l_c2w = calloc(sz_c2w, sizeof(float));
    float *l_c2b = calloc(CONV2_FILTERS, sizeof(float));
    float *l_c3w = calloc(sz_c3w, sizeof(float));
    float *l_c3b = calloc(CONV3_FILTERS, sizeof(float));
    float *l_fc1w = calloc(sz_fc1w, sizeof(float));
    float *l_fc1b = calloc(HIDDEN_NODES, sizeof(float));
    float *l_fc2w = calloc(sz_fc2w, sizeof(float));
    float *l_fc2b = calloc(OUTPUT_NODES, sizeof(float));

    // Error buffers
    float *d_p3 = malloc(FLATTEN_SIZE * sizeof(float));
    float *d_c3 =
        malloc(CONV3_SIZE * CONV3_SIZE * CONV3_FILTERS * sizeof(float));
    float *d_p2 =
        malloc(POOL2_SIZE * POOL2_SIZE * CONV2_FILTERS * sizeof(float));
    float *d_c2 =
        malloc(CONV2_SIZE * CONV2_SIZE * CONV2_FILTERS * sizeof(float));
    float *d_p1 =
        malloc(POOL1_SIZE * POOL1_SIZE * CONV1_FILTERS * sizeof(float));
    float *d_c1 =
        malloc(CONV1_SIZE * CONV1_SIZE * CONV1_FILTERS * sizeof(float));

    float local_loss = 0.0f;

#pragma omp for
    for (int b = 0; b < batch_size; b++) {
      float *in = &batch_input[b * 784];
      float *tgt = &batch_target[b * OUTPUT_NODES];

      // Forward
      nn_forward_impl(nn, in, t_c1, t_p1, t_c2, t_p2, t_c3, t_p3, t_fc1, t_out);

      for (int i = 0; i < OUTPUT_NODES; i++)
        if (tgt[i] > 0.5f)
          local_loss -= logf(t_out[i] + 1e-9f);

      // Backprop FC2
      float err_out[OUTPUT_NODES];
      for (int i = 0; i < OUTPUT_NODES; i++) {
        err_out[i] = t_out[i] - tgt[i];
        l_fc2b[i] += err_out[i];
        for (int h = 0; h < HIDDEN_NODES; h++)
          l_fc2w[h * OUTPUT_NODES + i] += err_out[i] * t_fc1[h];
      }

      // Backprop FC1
      memset(d_p3, 0, FLATTEN_SIZE * sizeof(float));
      for (int h = 0; h < HIDDEN_NODES; h++) {
        float err = 0.0f;
        for (int o = 0; o < OUTPUT_NODES; o++)
          err += err_out[o] * nn->fc2_w[h * OUTPUT_NODES + o];
        err *= d_relu(t_fc1[h]);
        l_fc1b[h] += err;
        for (int i = 0; i < FLATTEN_SIZE; i++) {
          l_fc1w[i * HIDDEN_NODES + h] += err * t_p3[i];
          d_p3[i] += err * nn->fc1_w[i * HIDDEN_NODES + h];
        }
      }

      // Backprop Conv3
      memset(d_c3, 0, CONV3_SIZE * CONV3_SIZE * CONV3_FILTERS * sizeof(float));
      backprop_pool(d_p3, d_c3, t_c3, CONV3_SIZE, CONV3_FILTERS);
      for (int i = 0; i < CONV3_SIZE * CONV3_SIZE * CONV3_FILTERS; i++)
        d_c3[i] *= d_relu(t_c3[i]);

      memset(d_p2, 0, POOL2_SIZE * POOL2_SIZE * CONV2_FILTERS * sizeof(float));
      for (int f = 0; f < CONV3_FILTERS; f++) {
        for (int y = 0; y < CONV3_SIZE; y++) {
          for (int x = 0; x < CONV3_SIZE; x++) {
            float g = d_c3[f * CONV3_SIZE * CONV3_SIZE + y * CONV3_SIZE + x];
            l_c3b[f] += g;
            for (int if_idx = 0; if_idx < CONV2_FILTERS; if_idx++) {
              int w_off = f * (CONV2_FILTERS * 9) + if_idx * 9;
              int in_off = if_idx * (POOL2_SIZE * POOL2_SIZE);
              for (int ky = 0; ky < 3; ky++) {
                int iy = y + ky - 1;
                if (iy >= 0 && iy < 7) {
                  for (int kx = 0; kx < 3; kx++) {
                    int ix = x + kx - 1;
                    if (ix >= 0 && ix < 7) {
                      float val = t_p2[in_off + iy * 7 + ix];
                      l_c3w[w_off + ky * 3 + kx] += g * val;
                      d_p2[in_off + iy * 7 + ix] +=
                          g * nn->c3_w[w_off + ky * 3 + kx];
                    }
                  }
                }
              }
            }
          }
        }
      }

      // Backprop Conv2
      memset(d_c2, 0, CONV2_SIZE * CONV2_SIZE * CONV2_FILTERS * sizeof(float));
      backprop_pool(d_p2, d_c2, t_c2, CONV2_SIZE, CONV2_FILTERS);
      for (int i = 0; i < CONV2_SIZE * CONV2_SIZE * CONV2_FILTERS; i++)
        d_c2[i] *= d_relu(t_c2[i]);

      memset(d_p1, 0, POOL1_SIZE * POOL1_SIZE * CONV1_FILTERS * sizeof(float));
      for (int f = 0; f < CONV2_FILTERS; f++) {
        for (int y = 0; y < CONV2_SIZE; y++) {
          for (int x = 0; x < CONV2_SIZE; x++) {
            float g = d_c2[f * CONV2_SIZE * CONV2_SIZE + y * CONV2_SIZE + x];
            l_c2b[f] += g;
            for (int if_idx = 0; if_idx < CONV1_FILTERS; if_idx++) {
              int w_off = f * (CONV1_FILTERS * 9) + if_idx * 9;
              int in_off = if_idx * (POOL1_SIZE * POOL1_SIZE);
              for (int ky = 0; ky < 3; ky++) {
                int iy = y + ky - 1;
                if (iy >= 0 && iy < 14) {
                  for (int kx = 0; kx < 3; kx++) {
                    int ix = x + kx - 1;
                    if (ix >= 0 && ix < 14) {
                      float val = t_p1[in_off + iy * 14 + ix];
                      l_c2w[w_off + ky * 3 + kx] += g * val;
                      d_p1[in_off + iy * 14 + ix] +=
                          g * nn->c2_w[w_off + ky * 3 + kx];
                    }
                  }
                }
              }
            }
          }
        }
      }

      // Backprop Conv1
      memset(d_c1, 0, CONV1_SIZE * CONV1_SIZE * CONV1_FILTERS * sizeof(float));
      backprop_pool(d_p1, d_c1, t_c1, CONV1_SIZE, CONV1_FILTERS);
      for (int i = 0; i < CONV1_SIZE * CONV1_SIZE * CONV1_FILTERS; i++)
        d_c1[i] *= d_relu(t_c1[i]);

      for (int f = 0; f < CONV1_FILTERS; f++) {
        for (int y = 0; y < CONV1_SIZE; y++) {
          for (int x = 0; x < CONV1_SIZE; x++) {
            float g = d_c1[f * CONV1_SIZE * CONV1_SIZE + y * CONV1_SIZE + x];
            l_c1b[f] += g;
            int w_off = f * 9;
            for (int ky = 0; ky < 3; ky++) {
              int iy = y + ky - 1;
              if (iy >= 0 && iy < 28) {
                for (int kx = 0; kx < 3; kx++) {
                  int ix = x + kx - 1;
                  if (ix >= 0 && ix < 28) {
                    float val = in[iy * 28 + ix];
                    l_c1w[w_off + ky * 3 + kx] += g * val;
                  }
                }
              }
            }
          }
        }
      }
    }

#pragma omp critical
    {
      total_loss += local_loss;
      for (int i = 0; i < sz_fc2w; i++)
        g_fc2w[i] += l_fc2w[i];
      for (int i = 0; i < OUTPUT_NODES; i++)
        g_fc2b[i] += l_fc2b[i];
      for (int i = 0; i < sz_fc1w; i++)
        g_fc1w[i] += l_fc1w[i];
      for (int i = 0; i < HIDDEN_NODES; i++)
        g_fc1b[i] += l_fc1b[i];
      for (int i = 0; i < sz_c3w; i++)
        g_c3w[i] += l_c3w[i];
      for (int i = 0; i < CONV3_FILTERS; i++)
        g_c3b[i] += l_c3b[i];
      for (int i = 0; i < sz_c2w; i++)
        g_c2w[i] += l_c2w[i];
      for (int i = 0; i < CONV2_FILTERS; i++)
        g_c2b[i] += l_c2b[i];
      for (int i = 0; i < sz_c1w; i++)
        g_c1w[i] += l_c1w[i];
      for (int i = 0; i < CONV1_FILTERS; i++)
        g_c1b[i] += l_c1b[i];
    }
    free(t_c1);
    free(t_p1);
    free(t_c2);
    free(t_p2);
    free(t_c3);
    free(t_p3);
    free(t_fc1);
    free(t_out);
    free(l_fc2w);
    free(l_fc2b);
    free(l_fc1w);
    free(l_fc1b);
    free(l_c3w);
    free(l_c3b);
    free(l_c2w);
    free(l_c2b);
    free(l_c1w);
    free(l_c1b);
    free(d_p3);
    free(d_c3);
    free(d_p2);
    free(d_c2);
    free(d_p1);
    free(d_c1);
  }

  float scale = 1.0f / batch_size;
  for (int i = 0; i < sz_fc2w; i++)
    g_fc2w[i] *= scale;
  for (int i = 0; i < OUTPUT_NODES; i++)
    g_fc2b[i] *= scale;
  for (int i = 0; i < sz_fc1w; i++)
    g_fc1w[i] *= scale;
  for (int i = 0; i < HIDDEN_NODES; i++)
    g_fc1b[i] *= scale;
  for (int i = 0; i < sz_c3w; i++)
    g_c3w[i] *= scale;
  for (int i = 0; i < CONV3_FILTERS; i++)
    g_c3b[i] *= scale;
  for (int i = 0; i < sz_c2w; i++)
    g_c2w[i] *= scale;
  for (int i = 0; i < CONV2_FILTERS; i++)
    g_c2b[i] *= scale;
  for (int i = 0; i < sz_c1w; i++)
    g_c1w[i] *= scale;
  for (int i = 0; i < CONV1_FILTERS; i++)
    g_c1b[i] *= scale;

  adam(nn->fc2_w, nn->m_fc2_w, nn->v_fc2_w, g_fc2w, sz_fc2w, lr, nn->timestep);
  adam(nn->fc2_b, nn->m_fc2_b, nn->v_fc2_b, g_fc2b, OUTPUT_NODES, lr,
       nn->timestep);
  adam(nn->fc1_w, nn->m_fc1_w, nn->v_fc1_w, g_fc1w, sz_fc1w, lr, nn->timestep);
  adam(nn->fc1_b, nn->m_fc1_b, nn->v_fc1_b, g_fc1b, HIDDEN_NODES, lr,
       nn->timestep);
  adam(nn->c3_w, nn->m_c3_w, nn->v_c3_w, g_c3w, sz_c3w, lr, nn->timestep);
  adam(nn->c3_b, nn->m_c3_b, nn->v_c3_b, g_c3b, CONV3_FILTERS, lr,
       nn->timestep);
  adam(nn->c2_w, nn->m_c2_w, nn->v_c2_w, g_c2w, sz_c2w, lr, nn->timestep);
  adam(nn->c2_b, nn->m_c2_b, nn->v_c2_b, g_c2b, CONV2_FILTERS, lr,
       nn->timestep);
  adam(nn->c1_w, nn->m_c1_w, nn->v_c1_w, g_c1w, sz_c1w, lr, nn->timestep);
  adam(nn->c1_b, nn->m_c1_b, nn->v_c1_b, g_c1b, CONV1_FILTERS, lr,
       nn->timestep);

  free(g_fc2w);
  free(g_fc2b);
  free(g_fc1w);
  free(g_fc1b);
  free(g_c3w);
  free(g_c3b);
  free(g_c2w);
  free(g_c2b);
  free(g_c1w);
  free(g_c1b);
  return total_loss / batch_size;
}

void nn_save(NeuralNet *nn, const char *filename) {
  FILE *f = fopen(filename, "wb");
  if (!f)
    return;
  fwrite(nn->c1_w, sizeof(float), CONV1_FILTERS * 9, f);
  fwrite(nn->c1_b, sizeof(float), CONV1_FILTERS, f);
  fwrite(nn->c2_w, sizeof(float), CONV2_FILTERS * CONV1_FILTERS * 9, f);
  fwrite(nn->c2_b, sizeof(float), CONV2_FILTERS, f);
  fwrite(nn->c3_w, sizeof(float), CONV3_FILTERS * CONV2_FILTERS * 9, f);
  fwrite(nn->c3_b, sizeof(float), CONV3_FILTERS, f);
  fwrite(nn->fc1_w, sizeof(float), FLATTEN_SIZE * HIDDEN_NODES, f);
  fwrite(nn->fc1_b, sizeof(float), HIDDEN_NODES, f);
  fwrite(nn->fc2_w, sizeof(float), HIDDEN_NODES * OUTPUT_NODES, f);
  fwrite(nn->fc2_b, sizeof(float), OUTPUT_NODES, f);
  fclose(f);
}

NeuralNet *nn_load(const char *filename) {
  FILE *f = fopen(filename, "rb");
  if (!f)
    return NULL;
  NeuralNet *nn = nn_create();
  fread(nn->c1_w, sizeof(float), CONV1_FILTERS * 9, f);
  fread(nn->c1_b, sizeof(float), CONV1_FILTERS, f);
  fread(nn->c2_w, sizeof(float), CONV2_FILTERS * CONV1_FILTERS * 9, f);
  fread(nn->c2_b, sizeof(float), CONV2_FILTERS, f);
  fread(nn->c3_w, sizeof(float), CONV3_FILTERS * CONV2_FILTERS * 9, f);
  fread(nn->c3_b, sizeof(float), CONV3_FILTERS, f);
  fread(nn->fc1_w, sizeof(float), FLATTEN_SIZE * HIDDEN_NODES, f);
  fread(nn->fc1_b, sizeof(float), HIDDEN_NODES, f);
  fread(nn->fc2_w, sizeof(float), HIDDEN_NODES * OUTPUT_NODES, f);
  fread(nn->fc2_b, sizeof(float), OUTPUT_NODES, f);
  fclose(f);
  return nn;
}

NeuralNet *nn_clone(NeuralNet *src) {
  NeuralNet *dst = nn_create();

  int s_c1 = CONV1_FILTERS * 9;
  int s_c2 = CONV2_FILTERS * CONV1_FILTERS * 9;
  int s_c3 = CONV3_FILTERS * CONV2_FILTERS * 9;
  int s_fc1 = FLATTEN_SIZE * HIDDEN_NODES;
  int s_fc2 = HIDDEN_NODES * OUTPUT_NODES;

  memcpy(dst->c1_w, src->c1_w, s_c1 * sizeof(float));
  memcpy(dst->c1_b, src->c1_b, CONV1_FILTERS * sizeof(float));

  memcpy(dst->c2_w, src->c2_w, s_c2 * sizeof(float));
  memcpy(dst->c2_b, src->c2_b, CONV2_FILTERS * sizeof(float));

  memcpy(dst->c3_w, src->c3_w, s_c3 * sizeof(float));
  memcpy(dst->c3_b, src->c3_b, CONV3_FILTERS * sizeof(float));

  memcpy(dst->fc1_w, src->fc1_w, s_fc1 * sizeof(float));
  memcpy(dst->fc1_b, src->fc1_b, HIDDEN_NODES * sizeof(float));

  memcpy(dst->fc2_w, src->fc2_w, s_fc2 * sizeof(float));
  memcpy(dst->fc2_b, src->fc2_b, OUTPUT_NODES * sizeof(float));

  return dst;
}
