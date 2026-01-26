/**
 * @file image_proc.c
 * @brief Image Preprocessing and Canvas Operations
 *
 * This module handles image preprocessing operations required to convert user
 * drawing input into the format expected by the neural network. It performs
 * spatial downscaling from high-resolution canvas to 28x28 input size, computes
 * center-of-mass for image centering, and applies brush strokes to the drawing
 * canvas with anti-aliased edges.
 */

#include "image_proc.h"
#include "app_state.h"
#include <math.h>
#include <string.h>

/**
 * Downscales high-resolution canvas input to 28x28 network input size.
 *
 * Performs area-averaging downsampling from the high-resolution canvas
 * (280x280) to the network's expected input dimensions (28x28). Each output
 * pixel is computed as the mean of the corresponding 10x10 block in the source
 * image, ensuring that the downscaled representation preserves the overall
 * intensity distribution of the original drawing.
 *
 * @param high_res Pointer to the high-resolution canvas buffer (CANVAS_DIM x
 * CANVAS_DIM)
 * @param low_res Pointer to the output buffer (28 x 28), must be pre-allocated
 */
void downscale_input(float *high_res, float *low_res) {
  for (int y = 0; y < 28; y++) {
    for (int x = 0; x < 28; x++) {
      float sum = 0;
      for (int dy = 0; dy < CANVAS_SCALE; dy++) {
        for (int dx = 0; dx < CANVAS_SCALE; dx++) {
          sum += high_res[(y * CANVAS_SCALE + dy) * CANVAS_DIM +
                          (x * CANVAS_SCALE + dx)];
        }
      }
      low_res[y * 28 + x] = sum / (float)(CANVAS_SCALE * CANVAS_SCALE);
    }
  }
}

/**
 * Centers the input image by computing and applying center-of-mass translation.
 *
 * Calculates the center of mass of the input image and shifts it so that the
 * centroid aligns with the center of the 28x28 grid (position 13.5, 13.5). This
 * normalization step improves classification accuracy by ensuring that digit
 * positions are consistent regardless of where they were drawn on the canvas.
 * If the input has insufficient mass (less than 0.1), the function copies the
 * input unchanged to avoid division by zero and preserve empty canvas state.
 *
 * @param src Pointer to the source 28x28 input buffer
 * @param dst Pointer to the output 28x28 buffer, must be pre-allocated
 */
void center_input(float *src, float *dst) {
  float sum_x = 0, sum_y = 0, total_mass = 0;
  for (int y = 0; y < 28; y++) {
    for (int x = 0; x < 28; x++) {
      float val = src[y * 28 + x];
      sum_x += x * val;
      sum_y += y * val;
      total_mass += val;
    }
  }
  if (total_mass < 0.1f) {
    memcpy(dst, src, 784 * sizeof(float));
    return;
  }
  float com_x = sum_x / total_mass;
  float com_y = sum_y / total_mass;
  int shift_x = (int)(13.5f - com_x);
  int shift_y = (int)(13.5f - com_y);
  memset(dst, 0, 784 * sizeof(float));
  for (int y = 0; y < 28; y++) {
    for (int x = 0; x < 28; x++) {
      int sx = x - shift_x;
      int sy = y - shift_y;
      if (sx >= 0 && sx < 28 && sy >= 0 && sy < 28)
        dst[y * 28 + x] = src[sy * 28 + sx];
    }
  }
}

/**
 * Applies a brush stroke to the high-resolution canvas at the specified
 * position.
 *
 * Renders a circular brush with a radius of 15 pixels at the given mouse
 * coordinates, with anti-aliased edges that fade from full intensity at the
 * center to zero at the radius boundary. The brush uses a 2-pixel transition
 * zone for smooth edges. Pixel values are accumulated additively and clamped to
 * a maximum of 1.0 to prevent overflow. This function is called during mouse
 * drag operations to build up the drawing incrementally.
 *
 * @param grid Pointer to the high-resolution canvas buffer (CANVAS_DIM x
 * CANVAS_DIM)
 * @param mx Mouse X coordinate in canvas space (0 to CANVAS_DIM-1)
 * @param my Mouse Y coordinate in canvas space (0 to CANVAS_DIM-1)
 */
void apply_brush_high_res(float *grid, int mx, int my) {
  float radius = 15.0f;
  for (int y = my - 15; y <= my + 15; y++) {
    for (int x = mx - 15; x <= mx + 15; x++) {
      if (x >= 0 && x < CANVAS_DIM && y >= 0 && y < CANVAS_DIM) {
        float d = sqrtf((x - mx) * (x - mx) + (y - my) * (y - my));
        if (d <= radius) {
          float val = (d > radius - 2.0f) ? (radius - d) / 2.0f : 1.0f;
          grid[y * CANVAS_DIM + x] =
              fminf(grid[y * CANVAS_DIM + x] + val, 1.0f);
        }
      }
    }
  }
}
