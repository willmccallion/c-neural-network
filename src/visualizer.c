/**
 * @file visualizer.c
 * @brief Neural Network Visualization and Rendering Utilities
 *
 * This module provides functions for rendering neural network internals,
 * including input images, layer activations, and network architecture
 * diagrams. It handles color interpolation for heatmap visualizations and
 * provides detailed views of convolutional feature maps and dense layer
 * activations for debugging and understanding network behavior.
 */

#include "visualizer.h"

/**
 * Linearly interpolates between two colors based on a parameter value.
 *
 * Computes a color that lies between color a and color b, with the
 * interpolation factor t determining the blend ratio. When t is 0, the result
 * equals a; when t is 1, the result equals b. The parameter is clamped to [0,
 * 1] to ensure valid color component values. The alpha channel is set to 255
 * (fully opaque). This is used for generating color gradients in heatmap
 * visualizations where activation intensity maps to color saturation.
 *
 * @param a Starting color (when t = 0)
 * @param b Ending color (when t = 1)
 * @param t Interpolation factor (clamped to [0, 1])
 * @return Interpolated color with alpha = 255
 */
Color LerpColor(Color a, Color b, float t) {
  if (t < 0.0f)
    t = 0.0f;
  if (t > 1.0f)
    t = 1.0f;
  return (Color){(unsigned char)(a.r + (b.r - a.r) * t),
                 (unsigned char)(a.g + (b.g - a.g) * t),
                 (unsigned char)(a.b + (b.b - a.b) * t), 255};
}

/**
 * Renders a 28x28 input image grid at the specified screen position.
 *
 * Draws each pixel of the input buffer as a scaled rectangle, with intensity
 * values mapped to grayscale colors. Pixels with values below 0.01 are skipped
 * to avoid rendering noise. The grid is surrounded by a border rectangle in
 * the specified color. This function is used to display both user-drawn input
 * and network-processed images in the visualization interface.
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
                     Color borderColor) {
  DrawRectangleLines(x - 1, y - 1, (28 * scale) + 2, (28 * scale) + 2,
                     borderColor);

  for (int i = 0; i < 28; i++) {
    for (int j = 0; j < 28; j++) {
      float val = input_buffer[j * 28 + i];
      if (val > 0.01f) {
        unsigned char v = (unsigned char)(val * 255.0f);
        DrawRectangle(x + i * scale, y + j * scale, scale, scale,
                      (Color){v, v, v, 255});
      }
    }
  }
}

/**
 * Renders a stack of 2D feature maps as offset overlapping visualizations.
 *
 * Displays up to 16 feature maps from a 3D tensor, with each map rendered at
 * a slight offset to create a stacked appearance. Each feature map is drawn
 * as a scaled grid where activation values are mapped to green color intensity
 * using linear interpolation. Maps with no significant activations (below 0.1)
 * are not rendered to reduce visual clutter. This visualization helps
 * understand what features each convolutional filter detects in the input.
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
                       int scale, const char *lbl) {
  DrawText(lbl, x, y - 25, 20, LIGHTGRAY);

  int limit = (d > 16) ? 16 : d;

  for (int i = 0; i < limit; i++) {
    int dx = x + (i * 6);
    int dy = y + (i * 6);

    DrawRectangle(dx, dy, w * scale, h * scale, (Color){20, 20, 20, 255});
    DrawRectangleLines(dx, dy, w * scale, h * scale, (Color){60, 60, 60, 255});

    for (int py = 0; py < h; py++) {
      for (int px = 0; px < w; px++) {
        float val = data[i * (w * h) + py * w + px];
        if (val > 0.1f) {
          Color c =
              LerpColor((Color){0, 50, 0, 255}, (Color){0, 255, 0, 255}, val);
          DrawRectangle(dx + px * scale, dy + py * scale, scale, scale, c);
        }
      }
    }
  }
  DrawText(TextFormat("%d Filters", d), x, y + h * scale + limit * 6 + 5, 10,
           GRAY);
}

/**
 * Renders a comprehensive visualization of the network's internal activations.
 *
 * Displays activation maps from all three convolutional layers and the dense
 * hidden layer, providing a complete view of how the network processes input
 * through its hierarchical feature extraction stages. Each layer is rendered
 * with appropriate scaling and positioning to show the spatial reduction and
 * feature complexity progression. The dense layer is displayed as a 2D grid
 * of neurons with activation intensity mapped to color. This visualization
 * is used in the dashboard to help users understand what the network "sees"
 * at each processing stage.
 *
 * @param nn Pointer to the neural network structure (reads activation buffers)
 * @param x Screen X coordinate of the visualization panel's top-left corner
 * @param y Screen Y coordinate of the visualization panel's top-left corner
 * @param w Width of the visualization panel in pixels (unused, reserved for
 * layout)
 * @param h Height of the visualization panel in pixels (unused, reserved for
 * layout)
 */
void draw_network_detailed(NeuralNet *nn, int x, int y, int w, int h) {
  draw_tensor_stack(nn->c1_out, 28, 28, CONV1_FILTERS, x, y, 3,
                    "LAYER 1 (CONV)");

  draw_tensor_stack(nn->c2_out, 14, 14, CONV2_FILTERS, x + 200, y + 50, 4,
                    "LAYER 2 (CONV)");

  draw_tensor_stack(nn->c3_out, 7, 7, CONV3_FILTERS, x + 400, y + 100, 6,
                    "LAYER 3 (CONV)");

  int dx = x;
  int dy = y + 450;
  DrawText("DENSE LAYER (256 Neurons)", dx, dy - 25, 20, LIGHTGRAY);

  int cols = 32;
  int size = 8;
  int gap = 2;

  for (int i = 0; i < 256; i++) {
    int cx = i % cols;
    int cy = i / cols;
    float val = nn->fc1_out[i];

    Color c = (Color){30, 30, 30, 255};
    if (val > 0)
      c = LerpColor(c, (Color){0, 255, 255, 255}, val);

    DrawRectangle(dx + cx * (size + gap), dy + cy * (size + gap), size, size,
                  c);
  }
}
