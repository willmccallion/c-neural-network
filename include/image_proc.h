/**
 * @file image_proc.h
 * @brief Image Preprocessing Function Declarations
 *
 * This header declares functions for preprocessing user drawing input into the
 * format expected by the neural network, including spatial downscaling, image
 * centering, and canvas brush operations.
 */

#ifndef IMAGE_PROC_H
#define IMAGE_PROC_H

/**
 * Downscales high-resolution canvas input to 28x28 network input size.
 *
 * @param high_res Pointer to the high-resolution canvas buffer (CANVAS_DIM x
 * CANVAS_DIM)
 * @param low_res Pointer to the output buffer (28 x 28), must be pre-allocated
 */
void downscale_input(float *high_res, float *low_res);

/**
 * Centers the input image by computing and applying center-of-mass translation.
 *
 * @param src Pointer to the source 28x28 input buffer
 * @param dst Pointer to the output 28x28 buffer, must be pre-allocated
 */
void center_input(float *src, float *dst);

/**
 * Applies a brush stroke to the high-resolution canvas at the specified
 * position.
 *
 * @param grid Pointer to the high-resolution canvas buffer (CANVAS_DIM x
 * CANVAS_DIM)
 * @param mx Mouse X coordinate in canvas space (0 to CANVAS_DIM-1)
 * @param my Mouse Y coordinate in canvas space (0 to CANVAS_DIM-1)
 */
void apply_brush_high_res(float *grid, int mx, int my);

#endif
