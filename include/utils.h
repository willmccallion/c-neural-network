/**
 * @file utils.h
 * @brief Utility Function Declarations
 *
 * This header declares helper functions for file system operations, path
 * resolution, and label mapping between class indices and human-readable
 * strings.
 */

#ifndef UTILS_H
#define UTILS_H

#include <stdbool.h>

/**
 * Checks whether a file exists at the specified path.
 *
 * @param path File system path to check
 * @return true if the file exists and is readable, false otherwise
 */
bool file_exists(const char *path);

/**
 * Resolves a filename to a full path by searching common data directories.
 *
 * @param filename Base filename to search for
 * @return Newly allocated string containing the resolved path, or filename if
 * not found
 */
char *resolve_path(const char *filename);

/**
 * Maps a class index to its human-readable label string.
 *
 * @param index Class index (0 to OUTPUT_NODES-1)
 * @return Pointer to a static string containing the label, or "???" for invalid
 * indices
 */
const char *get_label(int index);

#endif
