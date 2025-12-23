#ifndef APP_STATE_H
#define APP_STATE_H

#include "nn.h"
#include "raylib.h"
#include <pthread.h>
#include <stdbool.h>

// Constants
#define MODEL_FILENAME "model.bin"
#define MAX_HISTORY 200
#define VIZ_UPDATE_RATE_TRAIN 0.1f
#define VIZ_UPDATE_RATE_IDLE 1.0f

#define CANVAS_SCALE 10
#define CANVAS_DIM (28 * CANVAS_SCALE)
#define DISPLAY_SCALE 2

// Colors
#define COL_BG (Color){10, 10, 14, 255}
#define COL_PANEL (Color){22, 22, 26, 255}
#define COL_ACCENT_1 (Color){0, 255, 200, 255}
#define COL_ACCENT_2 (Color){255, 0, 100, 255}
#define COL_TEXT_DIM (Color){100, 100, 110, 255}
#define COL_GRID (Color){40, 40, 50, 255}

typedef enum { TAB_DASHBOARD, TAB_TRAINING } AppTab;

typedef struct {
  // Thread control
  volatile bool run_training;
  volatile bool should_quit;
  volatile bool new_best_available;
  pthread_mutex_t data_lock;

  // Training stats
  volatile int epoch;
  volatile int max_epochs;
  volatile int current_batch;
  volatile int total_batches;
  volatile float train_loss;
  volatile float val_accuracy;
  volatile float best_accuracy;

  // History
  float history_loss[MAX_HISTORY];
  float history_acc[MAX_HISTORY];
  int history_idx;

  // Visualization
  float viz_image[784];
  float viz_probs[OUTPUT_NODES];
  int viz_target_label;
  int viz_pred_label;

  // Models
  NeuralNet *gui_nn;   // Stable model for Dashboard
  NeuralNet *train_nn; // Mutating model for Training
  AppTab current_tab;
} AppState;

// Declare global instance
extern AppState appState;

#endif
