/**
 * @file utils.c
 * @brief Utility Functions for File Operations and Label Mapping
 *
 * This module provides helper functions for file system operations, path
 * resolution across different build configurations, and mapping between
 * class indices and human-readable label strings. These utilities abstract
 * away platform-specific path handling and provide consistent interfaces
 * for accessing data files and displaying classification results.
 */

#include "utils.h"
#include <stdio.h>
#include <string.h>

/**
 * Checks whether a file exists at the specified path.
 *
 * Attempts to open the file in read mode and immediately closes it if
 * successful. This is a lightweight existence check that does not verify file
 * permissions or contents. Returns false if the file cannot be opened, which
 * may indicate either a missing file or insufficient permissions.
 *
 * @param path File system path to check
 * @return true if the file exists and is readable, false otherwise
 */
bool file_exists(const char *path) {
  FILE *f = fopen(path, "r");
  if (f) {
    fclose(f);
    return true;
  }
  return false;
}

/**
 * Resolves a filename to a full path by searching common data directories.
 *
 * Searches for the specified filename in a sequence of standard locations:
 * current directory, data/, ../data/, build/data/, and build/. This allows
 * the application to find data files regardless of whether it is run from
 * the project root, build directory, or other locations. Returns a newly
 * allocated string containing the first matching path found, or the original
 * filename if no match is found. The caller is responsible for freeing the
 * returned string.
 *
 * @param filename Base filename to search for
 * @return Newly allocated string containing the resolved path, or filename if
 * not found
 */
char *resolve_path(const char *filename) {
  char path[256];

  sprintf(path, "%s", filename);
  if (file_exists(path))
    return strdup(path);

  sprintf(path, "data/%s", filename);
  if (file_exists(path))
    return strdup(path);

  sprintf(path, "../data/%s", filename);
  if (file_exists(path))
    return strdup(path);

  sprintf(path, "build/data/%s", filename);
  if (file_exists(path))
    return strdup(path);

  sprintf(path, "build/%s", filename);
  if (file_exists(path))
    return strdup(path);

  return strdup(filename);
}

/**
 * Maps a class index to its human-readable label string.
 *
 * Converts a numeric class index (0-69) into a string representation suitable
 * for display. The mapping covers digits (0-9), uppercase letters (A-Z),
 * lowercase letters (a, b, d, e, f, g, h, n, q, r, t), and drawing categories
 * (apple, book, candle, etc.). The function uses a static buffer for the return
 * value, so the string should be copied if it needs to persist across multiple
 * calls. Returns "???" for invalid indices outside the supported range.
 *
 * @param index Class index (0 to OUTPUT_NODES-1)
 * @return Pointer to a static string containing the label, or "???" for invalid
 * indices
 */
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
