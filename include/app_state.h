/**
 * @file app_state.h
 * @brief Application State Structure and Shared Constants
 *
 * This header defines the central application state structure that coordinates
 * communication between the main rendering thread and the background training
 * thread. It includes training statistics, visualization buffers, neural
 * network instances, and synchronization primitives. The module also defines UI
 * constants, color schemes, and canvas dimensions used throughout the
 * application.
 */

#ifndef APP_STATE_H
#define APP_STATE_H

#include "nn.h"
#include "raylib.h"
#include <pthread.h>
#include <stdbool.h>

/**
 * Filename for the persisted neural network model checkpoint.
 */
#define MODEL_FILENAME "model.bin"

/**
 * Maximum number of data points stored in the training history buffer.
 */
#define MAX_HISTORY 200

/**
 * Time interval (in seconds) between visualization updates during active
 * training.
 */
#define VIZ_UPDATE_RATE_TRAIN 0.1f

/**
 * Time interval (in seconds) between visualization updates during idle
 * evaluation mode.
 */
#define VIZ_UPDATE_RATE_IDLE 1.0f

/**
 * Scaling factor from network input size (28x28) to high-resolution canvas
 * dimensions.
 */
#define CANVAS_SCALE 10

/**
 * High-resolution canvas dimension (280x280 pixels) for user drawing input.
 */
#define CANVAS_DIM (28 * CANVAS_SCALE)

/**
 * Pixel scaling factor for displaying the canvas on screen (each canvas pixel
 * becomes 2x2 screen pixels).
 */
#define DISPLAY_SCALE 2

/**
 * Background color for the main application window.
 */
#define COL_BG (Color){10, 10, 14, 255}

/**
 * Panel background color for UI components.
 */
#define COL_PANEL (Color){22, 22, 26, 255}

/**
 * Primary accent color used for active elements and positive indicators.
 */
#define COL_ACCENT_1 (Color){0, 255, 200, 255}

/**
 * Secondary accent color used for loss values and warning indicators.
 */
#define COL_ACCENT_2 (Color){255, 0, 100, 255}

/**
 * Dimmed text color for inactive or secondary UI elements.
 */
#define COL_TEXT_DIM (Color){100, 100, 110, 255}

/**
 * Grid and border color for UI component outlines.
 */
#define COL_GRID (Color){40, 40, 50, 255}

/**
 * Application tab enumeration for UI navigation.
 *
 * Defines the available top-level views in the application interface. The
 * dashboard tab provides interactive drawing and real-time inference, while the
 * training analytics tab displays training progress, layer visualizations, and
 * weight distributions.
 */
typedef enum { TAB_DASHBOARD, TAB_TRAINING } AppTab;

/**
 * Central application state structure for thread coordination.
 *
 * This structure holds all shared state between the main rendering thread and
 * the background training thread. It includes volatile flags for thread
 * control, a mutex for protecting concurrent access to shared data, training
 * statistics, visualization buffers, and references to the neural network
 * instances. The structure is designed to allow the training thread to update
 * statistics and visualization data while the main thread reads them for
 * display, with proper synchronization to prevent race conditions.
 */
typedef struct {
  /**
   * Flag indicating whether the training thread should actively train the
   * network. When false, the thread enters idle evaluation mode. Modified by
   * the main thread via the pause/resume button, read by the training thread.
   */
  volatile bool run_training;

  /**
   * Flag signaling the training thread to exit and terminate. Set by the main
   * thread during application shutdown to gracefully stop the background
   * training loop.
   */
  volatile bool should_quit;

  /**
   * Flag indicating that a new best model checkpoint has been saved and the GUI
   * network should be reloaded. Set by the training thread when validation
   * accuracy improves, cleared by the main thread after reloading the model.
   */
  volatile bool new_best_available;

  /**
   * Mutex protecting access to shared data structures, including visualization
   * buffers, history arrays, and statistics. Must be held when reading or
   * writing any non-volatile fields that are accessed by both threads.
   */
  pthread_mutex_t data_lock;

  /**
   * Current training epoch number (1-indexed). Updated by the training thread,
   * read by the main thread for display.
   */
  volatile int epoch;

  /**
   * Maximum number of epochs to train. Set during training thread
   * initialization, read by both threads to determine training completion.
   */
  volatile int max_epochs;

  /**
   * Current batch index within the current epoch (0-indexed). Updated by the
   * training thread during batch processing, used for progress visualization.
   */
  volatile int current_batch;

  /**
   * Total number of batches per epoch. Calculated from dataset size and batch
   * size, used to compute training progress percentage.
   */
  volatile int total_batches;

  /**
   * Exponential moving average of training loss. Updated by the training thread
   * after each batch with a 95% retention factor, providing smoothed loss
   * visualization.
   */
  volatile float train_loss;

  /**
   * Most recently computed validation accuracy (0.0 to 1.0). Updated by the
   * training thread after each epoch's validation pass, used for accuracy
   * tracking and checkpoint selection.
   */
  volatile float val_accuracy;

  /**
   * Best validation accuracy achieved so far. Updated by the training thread
   * when a new best is found, triggers model checkpoint saving. Initialized to
   * 0.0.
   */
  volatile float best_accuracy;

  /**
   * Circular buffer storing training loss values for historical visualization.
   * Maintains the most recent MAX_HISTORY loss measurements, with older values
   * shifted out when the buffer is full. Protected by data_lock.
   */
  float history_loss[MAX_HISTORY];

  /**
   * Circular buffer storing validation accuracy values for historical
   * visualization. Maintains the most recent MAX_HISTORY accuracy measurements,
   * synchronized with history_loss. Protected by data_lock.
   */
  float history_acc[MAX_HISTORY];

  /**
   * Current write index into the history buffers, or count of valid entries if
   * less than MAX_HISTORY. Incremented by the training thread, read by the main
   * thread for graph rendering. Protected by data_lock.
   */
  int history_idx;

  /**
   * Current image being displayed in the visualization feed (28x28, normalized
   * 0-1). Updated by the training thread from the current batch or validation
   * samples, read by the main thread for rendering. Protected by data_lock.
   */
  float viz_image[784];

  /**
   * Output class probabilities for the current visualization image.
   * Updated by the training thread after forward propagation, read by the main
   * thread for displaying top predictions. Protected by data_lock.
   */
  float viz_probs[OUTPUT_NODES];

  /**
   * Ground truth label index for the current visualization image.
   * Set by the training thread from the dataset, used for displaying correct
   * classification targets. Protected by data_lock.
   */
  int viz_target_label;

  /**
   * Predicted label index (class with highest probability) for the current
   * visualization image. Computed by the training thread, used for displaying
   * network predictions. Protected by data_lock.
   */
  int viz_pred_label;

  /**
   * Stable neural network instance used for GUI inference operations.
   * This network is only updated when a new best model is available, ensuring
   * consistent predictions during user interaction. Owned and managed by the
   * main thread, read-only access from training thread.
   */
  NeuralNet *gui_nn;

  /**
   * Training neural network instance that undergoes weight updates during
   * training. This network is continuously modified by the training thread and
   * should not be used for GUI inference to avoid displaying inconsistent
   * intermediate states. Owned and managed by the training thread.
   */
  NeuralNet *train_nn;

  /**
   * Currently active UI tab (dashboard or training analytics).
   * Modified by the main thread based on user input, determines which view is
   * rendered in the main loop.
   */
  AppTab current_tab;
} AppState;

/**
 * Global application state instance.
 *
 * This is the single instance of AppState used throughout the application.
 * It is defined in main.c and declared here for access from other modules.
 * All access to this structure must respect the data_lock mutex for thread
 * safety.
 */
extern AppState appState;

#endif
