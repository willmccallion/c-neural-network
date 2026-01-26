/**
 * @file trainer.h
 * @brief Background Training Thread Function Declarations
 *
 * This header declares the training thread entry point and history management
 * function. The training thread operates independently from the main rendering
 * loop and coordinates through the shared AppState structure.
 */

#ifndef TRAINER_H
#define TRAINER_H

/**
 * Background training thread entry point.
 *
 * @param arg Thread argument (unused, always NULL)
 * @return NULL (thread exit value, unused)
 */
void *train_thread(void *arg);

/**
 * Appends loss and accuracy values to the training history buffer.
 *
 * @param loss Cross-entropy loss value to record
 * @param acc Validation accuracy value to record (0.0 to 1.0)
 */
void push_history(float loss, float acc);

#endif
