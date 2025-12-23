#ifndef TRAINER_H
#define TRAINER_H

void *train_thread(void *arg);
void push_history(float loss, float acc);

#endif
