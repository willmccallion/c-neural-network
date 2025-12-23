#include "image_proc.h"
#include "app_state.h"
#include <math.h>
#include <string.h>

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
