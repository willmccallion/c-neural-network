#include "utils.h"
#include <stdio.h>
#include <string.h>

bool file_exists(const char *path) {
  FILE *f = fopen(path, "r");
  if (f) {
    fclose(f);
    return true;
  }
  return false;
}

char *resolve_path(const char *filename) {
  char path[256];

  // Check current directory
  sprintf(path, "%s", filename);
  if (file_exists(path))
    return strdup(path);

  // Check data/
  sprintf(path, "data/%s", filename);
  if (file_exists(path))
    return strdup(path);

  // Check ../data/
  sprintf(path, "../data/%s", filename);
  if (file_exists(path))
    return strdup(path);

  // Check build/data/
  sprintf(path, "build/data/%s", filename);
  if (file_exists(path))
    return strdup(path);

  // Check build/
  sprintf(path, "build/%s", filename);
  if (file_exists(path))
    return strdup(path);

  return strdup(filename);
}

const char *get_label(int index) {
  static char buf[32];
  if (index < 10) {
    sprintf(buf, "%d", index);
    return buf;
  }
  if (index < 36) {
    sprintf(buf, "%c", 'A' + (index - 10));
    return buf;
  }
  if (index < 47) {
    const char map[] = {'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'};
    sprintf(buf, "%c", map[index - 36]);
    return buf;
  }
  const char *drawings[] = {
      "APPLE",     "BOOK", "CANDLE",    "CLOUD",  "CUP",      "DOOR",
      "ENVELOPE",  "EYE",  "FISH",      "GUITAR", "HAMMER",   "HAT",
      "ICE CREAM", "LEAF", "LIGHTNING", "MOON",   "MOUNTAIN", "STAR",
      "TENT",      "TREE", "UMBRELLA",  "WHEEL"};

  int drawing_idx = index - 47;
  int num_drawings = sizeof(drawings) / sizeof(drawings[0]);
  if (drawing_idx >= 0 && drawing_idx < num_drawings)
    return drawings[drawing_idx];
  return "???";
}
