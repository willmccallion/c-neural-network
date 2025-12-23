#include "gui.h"
#include "app_state.h"
#include "raylib.h"
#include "utils.h"
#include "visualizer.h"

void DrawHistogram(int x, int y, int w, int h, float *data, int count,
                   const char *title, float range) {
  DrawRectangle(x, y, w, h, COL_PANEL);
  DrawRectangleLines(x, y, w, h, COL_GRID);
  DrawText(title, x + 10, y + 10, 10, GRAY);

  int bins[40] = {0};
  int max_bin_count = 0;
  int step = (count > 2000) ? count / 2000 : 1;

  for (int i = 0; i < count; i += step) {
    float val = data[i];
    if (val < -range)
      val = -range;
    if (val > range)
      val = range;
    float norm = (val + range) / (2.0f * range);
    int bin = (int)(norm * 39.0f);
    if (bin < 0)
      bin = 0;
    if (bin > 39)
      bin = 39;
    bins[bin]++;
    if (bins[bin] > max_bin_count)
      max_bin_count = bins[bin];
  }
  if (max_bin_count == 0)
    max_bin_count = 1;

  int bar_w = (w - 20) / 40;
  for (int i = 0; i < 40; i++) {
    int bar_h = (int)((float)bins[i] / max_bin_count * (h - 40));
    DrawRectangle(x + 10 + i * bar_w, y + h - 10 - bar_h, bar_w - 1, bar_h,
                  COL_ACCENT_1);
  }
}

void DrawDualGraph(int x, int y, int w, int h, float *d1, float *d2,
                   int count) {
  DrawRectangle(x, y, w, h, COL_PANEL);
  DrawRectangleLines(x, y, w, h, COL_GRID);
  for (int i = 1; i < 5; i++)
    DrawLine(x, y + (h / 5) * i, x + w, y + (h / 5) * i,
             (Color){30, 30, 35, 255});

  if (count < 2)
    return;
  for (int i = 0; i < count - 1; i++) {
    int x1 = x + (int)((float)i / (MAX_HISTORY - 1) * w);
    int x2 = x + (int)((float)(i + 1) / (MAX_HISTORY - 1) * w);
    int ly1 = y + h - (int)((d1[i] / 2.5f) * (h - 20)) - 10;
    int ly2 = y + h - (int)((d1[i + 1] / 2.5f) * (h - 20)) - 10;
    DrawLine(x1, ly1, x2, ly2, COL_ACCENT_2);
    int ay1 = y + h - (int)(d2[i] * (h - 20)) - 10;
    int ay2 = y + h - (int)(d2[i + 1] * (h - 20)) - 10;
    DrawLine(x1, ay1, x2, ay2, COL_ACCENT_1);
  }
  DrawText("LOSS", x + 10, y + 10, 10, COL_ACCENT_2);
  DrawText("ACCURACY", x + 10, y + 25, 10, COL_ACCENT_1);
}

void DrawLiveFeed(int x, int y, int w, int h) {
  DrawRectangle(x, y, w, h, COL_PANEL);
  DrawRectangleLines(x, y, w, h, COL_GRID);

  const char *title = appState.run_training ? "LIVE FEED (TRAINING)"
                                            : "MODEL EVALUATION (IDLE)";
  Color tc = appState.run_training ? WHITE : COL_ACCENT_1;
  DrawText(title, x + 15, y + 15, 20, tc);

  int img_scale = 8;
  int img_x = x + 20;
  int img_y = y + 50;

  // Draw the image currently in the buffer
  draw_input_grid(appState.viz_image, img_x, img_y, img_scale,
                  (Color){60, 60, 60, 255});

  int bar_x = img_x + (28 * img_scale) + 20;
  int bar_y = img_y;
  int bar_w = w - (bar_x - x) - 20;

  // Calculate top 5 from the *current* viz_probs
  int top_idx[5];
  float top_val[5];
  for (int k = 0; k < 5; k++) {
    top_val[k] = -1;
    top_idx[k] = 0;
  }
  for (int i = 0; i < OUTPUT_NODES; i++) {
    float v = appState.viz_probs[i];
    for (int k = 0; k < 5; k++) {
      if (v > top_val[k]) {
        for (int j = 4; j > k; j--) {
          top_val[j] = top_val[j - 1];
          top_idx[j] = top_idx[j - 1];
        }
        top_val[k] = v;
        top_idx[k] = i;
        break;
      }
    }
  }

  for (int i = 0; i < 5; i++) {
    int py = bar_y + (i * 45);
    DrawText(get_label(top_idx[i]), bar_x, py, 20, WHITE);
    DrawRectangle(bar_x, py + 22, bar_w, 8, (Color){30, 30, 30, 255});
    DrawRectangle(bar_x, py + 22, (int)(bar_w * top_val[i]), 8, COL_ACCENT_1);
    DrawText(TextFormat("%.1f%%", top_val[i] * 100), bar_x + bar_w - 45,
             py + 20, 10, GRAY);
  }
  DrawText(TextFormat("TARGET: %s", get_label(appState.viz_target_label)),
           img_x, img_y + 230, 20, GRAY);
}

void DrawLayerHeatmaps(int x, int y, int w, int h) {
  DrawRectangle(x, y, w, h, COL_PANEL);
  DrawRectangleLines(x, y, w, h, COL_GRID);
  DrawText("NETWORK INTERNALS (ACTIVATION MAPS)", x + 15, y + 15, 20, WHITE);

  int start_y = y + 50;

  // Layer 1: Edges (28x28)
  int c1_scale = 2;
  int c1_w = 28 * c1_scale;
  DrawText("L1: EDGES", x + 20, start_y - 20, 10, GRAY);
  for (int i = 0; i < 10; i++) {
    int dx = x + 20 + i * (c1_w + 5);
    DrawRectangleLines(dx - 1, start_y - 1, c1_w + 2, c1_w + 2,
                       (Color){40, 40, 40, 255});
    for (int py = 0; py < 28; py++) {
      for (int px = 0; px < 28; px++) {
        float val = appState.train_nn->c1_out[i * (28 * 28) + py * 28 + px];
        if (val > 0) {
          Color c = (Color){0, (unsigned char)(val * 255),
                            (unsigned char)(val * 200), 255};
          DrawRectangle(dx + px * c1_scale, start_y + py * c1_scale, c1_scale,
                        c1_scale, c);
        }
      }
    }
  }

  // Layer 2: Shapes (14x14)
  int c2_y = start_y + c1_w + 40;
  int c2_scale = 3;
  int c2_w = 14 * c2_scale;
  DrawText("L2: SHAPES", x + 20, c2_y - 20, 10, GRAY);
  for (int i = 0; i < 20; i++) {
    int dx = x + 20 + i * (c2_w + 5);
    DrawRectangleLines(dx - 1, c2_y - 1, c2_w + 2, c2_w + 2,
                       (Color){40, 40, 40, 255});
    for (int py = 0; py < 14; py++) {
      for (int px = 0; px < 14; px++) {
        float val = appState.train_nn->c2_out[i * (14 * 14) + py * 14 + px];
        if (val > 0) {
          Color c = (Color){(unsigned char)(val * 200),
                            (unsigned char)(val * 255), 0, 255};
          DrawRectangle(dx + px * c2_scale, c2_y + py * c2_scale, c2_scale,
                        c2_scale, c);
        }
      }
    }
  }

  // Layer 3: Concepts (7x7)
  int c3_y = c2_y + c2_w + 40;
  int c3_scale = 4;
  int c3_w = 7 * c3_scale;
  DrawText("L3: CONCEPTS", x + 20, c3_y - 20, 10, GRAY);
  for (int i = 0; i < 30; i++) {
    int dx = x + 20 + i * (c3_w + 5);
    DrawRectangleLines(dx - 1, c3_y - 1, c3_w + 2, c3_w + 2,
                       (Color){40, 40, 40, 255});
    for (int py = 0; py < 7; py++) {
      for (int px = 0; px < 7; px++) {
        float val = appState.train_nn->c3_out[i * (7 * 7) + py * 7 + px];
        if (val > 0) {
          Color c = (Color){(unsigned char)(val * 255), 100,
                            (unsigned char)(val * 100), 255};
          DrawRectangle(dx + px * c3_scale, c3_y + py * c3_scale, c3_scale,
                        c3_scale, c);
        }
      }
    }
  }
}

void DrawStatsPanel(int x, int y, int w, int h) {
  DrawRectangle(x, y, w, h, COL_PANEL);
  DrawRectangleLines(x, y, w, h, COL_GRID);

  // Button logic
  const char *btn_text =
      appState.run_training ? "PAUSE TRAINING" : "RESUME TRAINING";
  int text_w = MeasureText(btn_text, 20);
  int btn_w = text_w + 40;
  int btn_h = 40;
  int btn_x = x + 20;
  int btn_y = y + 20;

  Vector2 m = GetMousePosition();
  bool hover = (m.x > btn_x && m.x < btn_x + btn_w && m.y > btn_y &&
                m.y < btn_y + btn_h);
  Color btn_col = appState.run_training ? COL_ACCENT_2 : COL_ACCENT_1;
  if (hover)
    btn_col = ColorBrightness(btn_col, 0.2f);

  DrawRectangle(btn_x, btn_y, btn_w, btn_h, btn_col);
  DrawText(btn_text, btn_x + 20, btn_y + 10, 20, BLACK);

  if (hover && IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
    appState.run_training = !appState.run_training;
  }

  // Stats
  int col1_w = 300;
  int sy = btn_y + 60;
  DrawText("TRAINING STATUS", x + 20, sy, 20, WHITE);

  if (appState.run_training) {
    DrawText(TextFormat("EPOCH: %d / %d", appState.epoch, appState.max_epochs),
             x + 20, sy + 30, 20, COL_ACCENT_1);
    float progress = (float)appState.current_batch / appState.total_batches;
    DrawRectangle(x + 20, sy + 60, col1_w - 40, 6, (Color){40, 40, 40, 255});
    DrawRectangle(x + 20, sy + 60, (int)((col1_w - 40) * progress), 6,
                  COL_ACCENT_1);
    DrawText(TextFormat("LOSS: %.4f", appState.train_loss), x + 20, sy + 80, 20,
             COL_ACCENT_2);
  } else {
    DrawText("PAUSED (Idle Mode)", x + 20, sy + 30, 20, GRAY);
  }

  DrawText(TextFormat("BEST ACC: %.2f%%", appState.best_accuracy * 100), x + 20,
           sy + 110, 20, GREEN);

  int graph_x = x + col1_w;
  int graph_w = w - col1_w - 20;
  DrawDualGraph(graph_x, y + 20, graph_w, h - 40, appState.history_loss,
                appState.history_acc, appState.history_idx);
}
