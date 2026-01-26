/**
 * @file visualizer.h
 * @brief Neural Network Visualization Function Declarations
 *
 * This header declares functions for rendering neural network internals,
 * including input images, layer activations, and network architecture diagrams.
 */

#ifndef VISUALIZER_H
#define VISUALIZER_H

#include "nn.h"
#include "raylib.h"

/**
 * Linearly interpolates between two colors based on a parameter value.
 *
 * @param a Starting color (when t = 0)
 * @param b Ending color (when t = 1)
 * @param t Interpolation factor (clamped to [0, 1])
 * @return Interpolated color with alpha = 255
 */
Color LerpColor(Color a, Color b, float t);

/**
 * Renders a 28x28 input image grid at the specified screen position.
 *
 * @param input_buffer Pointer to the 28x28 input image (784 elements,
 * normalized 0-1)
 * @param x Screen X coordinate of the grid's top-left corner
 * @param y Screen Y coordinate of the grid's top-left corner
 * @param scale Pixel scaling factor (each input pixel becomes scale x scale
 * screen pixels)
 * @param borderColor Color for the border rectangle surrounding the grid
 */
void draw_input_grid(float *input_buffer, int x, int y, int scale,
                     Color borderColor);

/**
 * Renders a comprehensive visualization of the network's internal activations.
 *
 * @param nn Pointer to the neural network structure (reads activation buffers)
 * @param x Screen X coordinate of the visualization panel's top-left corner
 * @param y Screen Y coordinate of the visualization panel's top-left corner
 * @param w Width of the visualization panel in pixels (unused, reserved for
 * layout)
 * @param h Height of the visualization panel in pixels (unused, reserved for
 * layout)
 */
void draw_network_detailed(NeuralNet *nn, int x, int y, int w, int h);

/**
 * Renders a stack of 2D feature maps as offset overlapping visualizations.
 *
 * @param data Pointer to the 3D tensor data (d feature maps of size w x h)
 * @param w Width of each feature map
 * @param h Height of each feature map
 * @param d Depth (number of feature maps)
 * @param x Screen X coordinate of the visualization's top-left corner
 * @param y Screen Y coordinate of the visualization's top-left corner
 * @param scale Pixel scaling factor for rendering
 * @param lbl Text label to display above the visualization
 */
void draw_tensor_stack(float *data, int w, int h, int d, int x, int y,
                       int scale, const char *lbl);

#endif
