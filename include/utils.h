#ifndef UTILS_H
#define UTILS_H

#include <stdbool.h>

bool file_exists(const char *path);
char *resolve_path(const char *filename);
const char *get_label(int index);

#endif
