#ifndef VISUALIZER_H
#define VISUALIZER_H

#include "nn.h"
#include "raylib.h"

Color LerpColor(Color a, Color b, float t);

void draw_input_grid(float *input_buffer, int x, int y, int scale,
                     Color borderColor);

void draw_network_detailed(NeuralNet *nn, int x, int y, int w, int h);

void draw_tensor_stack(float *data, int w, int h, int d, int x, int y,
                       int scale, const char *lbl);

#endif
