#include "trainer.h"
#include "app_state.h"
#include "mnist.h"
#include "nn.h"
#include "raylib.h"
#include "utils.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

void push_history(float loss, float acc) {
  pthread_mutex_lock(&appState.data_lock);
  if (appState.history_idx < MAX_HISTORY) {
    appState.history_loss[appState.history_idx] = loss;
    appState.history_acc[appState.history_idx] = acc;
    appState.history_idx++;
  } else {
    for (int i = 1; i < MAX_HISTORY; i++) {
      appState.history_loss[i - 1] = appState.history_loss[i];
      appState.history_acc[i - 1] = appState.history_acc[i];
    }
    appState.history_loss[MAX_HISTORY - 1] = loss;
    appState.history_acc[MAX_HISTORY - 1] = acc;
  }
  pthread_mutex_unlock(&appState.data_lock);
}

void *train_thread(void *arg) {
  // Use separate malloc'd strings to avoid static buffer overwrite
  char *img_path = resolve_path("extended-train-images-idx3-ubyte");
  char *lbl_path = resolve_path("extended-train-labels-idx1-ubyte");

  MnistData *data = load_mnist(img_path, lbl_path);

  free(img_path);
  free(lbl_path);

  if (!data) {
    printf("CRITICAL ERROR: Data files not found.\n");
    // Create dummy data to prevent crash
    data = malloc(sizeof(MnistData));
    data->count = 100;
    data->width = 28;
    data->height = 28;
    data->images = malloc(100 * sizeof(unsigned char *));
    data->labels = malloc(100);
    for (int i = 0; i < 100; i++) {
      data->images[i] = calloc(784, 1);
      data->labels[i] = 0;
    }
  } else {
    printf("Loaded %d images.\n", data->count);
  }

  int train_count = (int)(data->count * 0.9);
  int val_count = data->count - train_count;
  int BATCH = 64;
  float *b_img = malloc(BATCH * 784 * sizeof(float));
  float *b_tgt = malloc(BATCH * OUTPUT_NODES * sizeof(float));
  float *inf_in = malloc(784 * sizeof(float));
  float *inf_out = malloc(OUTPUT_NODES * sizeof(float));

  appState.max_epochs = 200;
  appState.total_batches = train_count / BATCH;
  double last_viz_time = 0;

  // Check accuracy
  if (data && data->count > 0) {
    printf("Verifying loaded model accuracy...\n");
    int correct = 0;
    int startup_samples = 2000; // Check 2000 samples for a good estimate
    if (startup_samples > val_count)
      startup_samples = val_count;

    for (int i = 0; i < startup_samples; i++) {
      int idx = train_count + (rand() % val_count);
      for (int p = 0; p < 784; p++)
        inf_in[p] = data->images[idx][p] / 255.0f;

      nn_inference(appState.train_nn, inf_in, inf_out);

      int max_idx = 0;
      for (int k = 1; k < OUTPUT_NODES; k++)
        if (inf_out[k] > inf_out[max_idx])
          max_idx = k;
      if (max_idx == data->labels[idx])
        correct++;
    }
    appState.best_accuracy = (float)correct / startup_samples;
    printf("Verified Model Accuracy: %.2f%%\n",
           appState.best_accuracy * 100.0f);
  }

  while (!appState.should_quit) {

    // Mode 1: training
    if (appState.run_training) {
      for (int e = 0; e < appState.max_epochs && appState.run_training; e++) {
        appState.epoch = e + 1;

        for (int b = 0; b < appState.total_batches && appState.run_training;
             b++) {
          appState.current_batch = b;

          for (int k = 0; k < BATCH; k++) {
            int idx = rand() % train_count;
            for (int i = 0; i < 784; i++) {
              float val = data->images[idx][i] / 255.0f;
              if (rand() % 100 < 10)
                val += ((rand() % 100) / 400.0f); // Noise
              b_img[k * 784 + i] = val;
            }
            memset(&b_tgt[k * OUTPUT_NODES], 0, OUTPUT_NODES * 4);
            int lbl = data->labels[idx];
            if (lbl < OUTPUT_NODES)
              b_tgt[k * OUTPUT_NODES + lbl] = 1.0f;
          }

          float l =
              nn_train_batch(appState.train_nn, b_img, b_tgt, BATCH, 0.0005f);
          appState.train_loss = 0.95f * appState.train_loss + 0.05f * l;

          double now = GetTime();
          if (now - last_viz_time > VIZ_UPDATE_RATE_TRAIN) {
            last_viz_time = now;
            pthread_mutex_lock(&appState.data_lock);
            memcpy(appState.viz_image, &b_img[0], 784 * sizeof(float));

            // Use nn_forward to update internal layer buffers for visualization
            nn_forward(appState.train_nn, appState.viz_image, false);
            memcpy(appState.viz_probs, appState.train_nn->final_out,
                   OUTPUT_NODES * sizeof(float));

            int target = 0;
            for (int i = 0; i < OUTPUT_NODES; i++)
              if (b_tgt[i] > 0.5f)
                target = i;
            appState.viz_target_label = target;

            int max_p = 0;
            for (int i = 1; i < OUTPUT_NODES; i++)
              if (appState.viz_probs[i] > appState.viz_probs[max_p])
                max_p = i;
            appState.viz_pred_label = max_p;
            pthread_mutex_unlock(&appState.data_lock);
          }
        }

        if (appState.run_training) {
          int correct = 0;
          int val_samples = 500;
          if (val_samples > val_count)
            val_samples = val_count;

          for (int i = 0; i < val_samples; i++) {
            int idx = train_count + (rand() % val_count);
            for (int p = 0; p < 784; p++)
              inf_in[p] = data->images[idx][p] / 255.0f;

            nn_inference(appState.train_nn, inf_in, inf_out);

            int max_idx = 0;
            for (int k = 1; k < OUTPUT_NODES; k++)
              if (inf_out[k] > inf_out[max_idx])
                max_idx = k;
            if (max_idx == data->labels[idx])
              correct++;
          }
          appState.val_accuracy = (float)correct / val_samples;
          push_history(appState.train_loss, appState.val_accuracy);

          if (appState.val_accuracy > appState.best_accuracy) {
            appState.best_accuracy = appState.val_accuracy;
            nn_save(appState.train_nn, MODEL_FILENAME);
            appState.new_best_available = true;
          }
        }
      }
    }
    // Mode 2: idle
    else {
      double now = GetTime();
      if (now - last_viz_time > VIZ_UPDATE_RATE_IDLE) {
        last_viz_time = now;
        int idx = train_count + (rand() % val_count);

        pthread_mutex_lock(&appState.data_lock);
        // Load Image
        for (int p = 0; p < 784; p++)
          appState.viz_image[p] = data->images[idx][p] / 255.0f;
        appState.viz_target_label = data->labels[idx];

        // Use nn_forward so Idle mode also updates heatmaps
        nn_forward(appState.train_nn, appState.viz_image, false);
        memcpy(appState.viz_probs, appState.train_nn->final_out,
               OUTPUT_NODES * sizeof(float));

        int max_p = 0;
        for (int i = 1; i < OUTPUT_NODES; i++)
          if (appState.viz_probs[i] > appState.viz_probs[max_p])
            max_p = i;
        appState.viz_pred_label = max_p;

        pthread_mutex_unlock(&appState.data_lock);
      }
      usleep(50000);
    }
  }

  free(b_img);
  free(b_tgt);
  free(inf_in);
  free(inf_out);
  free_mnist(data);
  return NULL;
}
