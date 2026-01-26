/**
 * @file main.c
 * @brief Application Entry Point and Main Event Loop
 *
 * This module serves as the entry point for the MNIST drawing predictor
 * application. It initializes the neural network models, spawns the training
 * thread, and manages the main rendering loop that handles user input, model
 * inference, and visualization updates. The application maintains two separate
 * neural network instances: one for stable GUI inference and another for
 * concurrent training operations.
 */

#include "app_state.h"
#include "gui.h"
#include "image_proc.h"
#include "nn.h"
#include "raylib.h"
#include "trainer.h"
#include "utils.h"
#include "visualizer.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * Global application state instance.
 *
 * This structure holds all shared state between the main thread and the
 * training thread, including neural network instances, training statistics,
 * visualization buffers, and synchronization primitives. Access to this
 * structure is protected by the data_lock mutex for thread-safe operations.
 */
AppState appState = {0};

/**
 * Main application entry point.
 *
 * Initializes the application by loading or creating the neural network model,
 * spawning the background training thread, and entering the main rendering
 * loop. The function manages two separate neural network instances: gui_nn for
 * stable inference during user interaction, and train_nn for concurrent
 * training operations. The main loop handles user input on the drawing canvas,
 * performs real-time inference, updates visualizations, and coordinates model
 * updates from the training thread. The function blocks until the window is
 * closed, then performs cleanup of all allocated resources.
 *
 * @return Exit status code (0 on successful completion)
 */
int main() {
  pthread_mutex_init(&appState.data_lock, NULL);

  char *path = resolve_path(MODEL_FILENAME);
  appState.gui_nn = nn_load(path);

  if (!appState.gui_nn) {
    printf("No model found at %s. Creating NEW.\n", path);
    appState.gui_nn = nn_create();
    appState.run_training = true;
    appState.best_accuracy = 0.0f;
  } else {
    printf("Loaded model from %s.\n", path);
    appState.run_training = false;
    appState.best_accuracy = 0.0f;
  }
  free(path);

  appState.train_nn = nn_clone(appState.gui_nn);

  pthread_t t;
  pthread_create(&t, NULL, train_thread, NULL);

  InitWindow(1600, 950, "Drawing Predictor");
  SetTargetFPS(60);

  float *high_res_grid = calloc(CANVAS_DIM * CANVAS_DIM, sizeof(float));
  float *low_res_grid = calloc(784, sizeof(float));
  float *centered_grid = calloc(784, sizeof(float));

  int gx = 40;
  int gy = 80;
  int display_size = CANVAS_DIM * DISPLAY_SCALE;

  while (!WindowShouldClose()) {
    Vector2 m = GetMousePosition();

    if (appState.new_best_available) {
      pthread_mutex_lock(&appState.data_lock);
      appState.new_best_available = false;
      pthread_mutex_unlock(&appState.data_lock);

      nn_free(appState.gui_nn);
      appState.gui_nn = nn_load(MODEL_FILENAME);
      printf("GUI Model updated.\n");
    }

    if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON) && m.y < 50) {
      if (m.x < 200)
        appState.current_tab = TAB_DASHBOARD;
      else if (m.x < 400)
        appState.current_tab = TAB_TRAINING;
    }

    if (appState.current_tab == TAB_DASHBOARD) {
      if (IsMouseButtonDown(MOUSE_LEFT_BUTTON) && m.x > gx &&
          m.x < gx + display_size && m.y > gy && m.y < gy + display_size) {
        int mx = (int)((m.x - gx) / DISPLAY_SCALE);
        int my = (int)((m.y - gy) / DISPLAY_SCALE);
        apply_brush_high_res(high_res_grid, mx, my);
      }
      if (IsMouseButtonPressed(MOUSE_RIGHT_BUTTON) || IsKeyPressed(KEY_C))
        memset(high_res_grid, 0, CANVAS_DIM * CANVAS_DIM * sizeof(float));

      downscale_input(high_res_grid, low_res_grid);
      center_input(low_res_grid, centered_grid);
      nn_forward(appState.gui_nn, centered_grid, false);
    }

    BeginDrawing();
    ClearBackground(COL_BG);

    DrawRectangle(0, 0, 1600, 50, (Color){15, 15, 18, 255});
    Color t1 =
        (appState.current_tab == TAB_DASHBOARD) ? COL_ACCENT_1 : COL_TEXT_DIM;
    Color t2 =
        (appState.current_tab == TAB_TRAINING) ? COL_ACCENT_1 : COL_TEXT_DIM;
    DrawText("DASHBOARD", 40, 15, 20, t1);
    DrawText("TRAINING ANALYTICS", 220, 15, 20, t2);
    if (appState.current_tab == TAB_DASHBOARD)
      DrawRectangle(40, 45, 110, 3, t1);
    else
      DrawRectangle(220, 45, 190, 3, t2);

    if (appState.current_tab == TAB_DASHBOARD) {
      DrawText("INPUT CANVAS (High Res)", gx, gy - 30, 20, COL_TEXT_DIM);
      DrawRectangleLines(gx - 1, gy - 1, display_size + 2, display_size + 2,
                         (Color){60, 60, 70, 255});
      DrawRectangle(gx, gy, display_size, display_size,
                    (Color){10, 10, 10, 255});

      for (int y = 0; y < CANVAS_DIM; y += 1) {
        for (int x = 0; x < CANVAS_DIM; x += 1) {
          float val = high_res_grid[y * CANVAS_DIM + x];
          if (val > 0.05f) {
            unsigned char v = (unsigned char)(val * 255);
            DrawRectangle(gx + x * DISPLAY_SCALE, gy + y * DISPLAY_SCALE,
                          DISPLAY_SCALE, DISPLAY_SCALE, (Color){v, v, v, 255});
          }
        }
      }

      int robot_x = gx;
      int robot_y = gy + display_size + 40;
      DrawText("ROBOT VISION (28x28)", robot_x, robot_y - 25, 20, COL_ACCENT_1);
      draw_input_grid(centered_grid, robot_x, robot_y, 8, COL_ACCENT_1);

      int viz_x = 700;
      int viz_y = 80;
      DrawRectangle(viz_x - 20, viz_y - 20, 850, 850, COL_PANEL);
      draw_network_detailed(appState.gui_nn, viz_x, viz_y, 800, 800);

      int pred_x = viz_x + 500;
      int pred_y = viz_y + 450;
      DrawText("TOP PREDICTIONS", pred_x, pred_y, 20, COL_ACCENT_1);

      int top_idx[5];
      float top_val[5];
      for (int k = 0; k < 5; k++) {
        top_val[k] = -1;
        top_idx[k] = 0;
      }
      for (int i = 0; i < OUTPUT_NODES; i++) {
        float v = appState.gui_nn->final_out[i];
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
        int py = pred_y + 40 + (i * 50);
        DrawRectangle(pred_x, py + 25, 250, 10, (Color){30, 30, 30, 255});
        DrawRectangle(pred_x, py + 25, (int)(250 * top_val[i]), 10,
                      (i == 0 ? COL_ACCENT_1 : ORANGE));
        DrawText(get_label(top_idx[i]), pred_x, py, 20, WHITE);
        DrawText(TextFormat("%.1f%%", top_val[i] * 100), pred_x + 260, py + 20,
                 20, GRAY);
      }
    } else {
      int tx = 50;
      int ty = 80;
      DrawLiveFeed(tx, ty, 500, 280);
      DrawStatsPanel(tx + 520, ty, 980, 280);
      DrawLayerHeatmaps(tx, ty + 300, 1500, 280);

      int h_y = ty + 600;
      DrawText("LAYER WEIGHT DISTRIBUTIONS", tx, h_y, 20, WHITE);
      DrawHistogram(tx, h_y + 30, 360, 180, appState.train_nn->c1_w,
                    CONV1_FILTERS * 9, "Conv 1 (Range +/- 0.60)", 0.60f);
      DrawHistogram(tx + 380, h_y + 30, 360, 180, appState.train_nn->c2_w,
                    CONV2_FILTERS * CONV1_FILTERS * 9,
                    "Conv 2 (Range +/- 0.40)", 0.40f);
      DrawHistogram(tx + 760, h_y + 30, 360, 180, appState.train_nn->c3_w,
                    CONV3_FILTERS * CONV2_FILTERS * 9,
                    "Conv 3 (Range +/- 0.30)", 0.30f);
      DrawHistogram(tx + 1140, h_y + 30, 360, 180, appState.train_nn->fc1_w,
                    2000, "Dense (Range +/- 0.20)", 0.20f);
    }

    EndDrawing();
  }

  free(high_res_grid);
  free(low_res_grid);
  free(centered_grid);
  nn_free(appState.gui_nn);
  nn_free(appState.train_nn);
  pthread_mutex_destroy(&appState.data_lock);
  CloseWindow();
  return 0;
}
