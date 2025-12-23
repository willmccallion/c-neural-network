#include "visualizer.h"

Color LerpColor(Color a, Color b, float t) {
  if (t < 0.0f)
    t = 0.0f;
  if (t > 1.0f)
    t = 1.0f;
  return (Color){(unsigned char)(a.r + (b.r - a.r) * t),
                 (unsigned char)(a.g + (b.g - a.g) * t),
                 (unsigned char)(a.b + (b.b - a.b) * t), 255};
}

void draw_input_grid(float *input_buffer, int x, int y, int scale,
                     Color borderColor) {
  DrawRectangleLines(x - 1, y - 1, (28 * scale) + 2, (28 * scale) + 2,
                     borderColor);

  for (int i = 0; i < 28; i++) {
    for (int j = 0; j < 28; j++) {
      float val = input_buffer[j * 28 + i];
      if (val > 0.01f) {
        unsigned char v = (unsigned char)(val * 255.0f);
        // Draw pixel
        DrawRectangle(x + i * scale, y + j * scale, scale, scale,
                      (Color){v, v, v, 255});
      }
    }
  }
}

void draw_tensor_stack(float *data, int w, int h, int d, int x, int y,
                       int scale, const char *lbl) {
  DrawText(lbl, x, y - 25, 20, LIGHTGRAY);

  // Draw up to 16 filters to keep it clean
  int limit = (d > 16) ? 16 : d;

  for (int i = 0; i < limit; i++) {
    int dx = x + (i * 6);
    int dy = y + (i * 6);

    // Background of map
    DrawRectangle(dx, dy, w * scale, h * scale, (Color){20, 20, 20, 255});
    DrawRectangleLines(dx, dy, w * scale, h * scale, (Color){60, 60, 60, 255});

    // Draw content (subsampled)
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
  // Label depth
  DrawText(TextFormat("%d Filters", d), x, y + h * scale + limit * 6 + 5, 10,
           GRAY);
}

void draw_network_detailed(NeuralNet *nn, int x, int y, int w, int h) {
  // Conv Layer 1 (28x28x16)
  draw_tensor_stack(nn->c1_out, 28, 28, CONV1_FILTERS, x, y, 3,
                    "LAYER 1 (CONV)");

  // Conv Layer 2 (14x14x32)
  draw_tensor_stack(nn->c2_out, 14, 14, CONV2_FILTERS, x + 200, y + 50, 4,
                    "LAYER 2 (CONV)");

  // Conv Layer 3 (7x7x64)
  draw_tensor_stack(nn->c3_out, 7, 7, CONV3_FILTERS, x + 400, y + 100, 6,
                    "LAYER 3 (CONV)");

  // Dense Layer (Flattened view)
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
