/**
 * @file gui.h
 * @brief Graphical User Interface Rendering Function Declarations
 *
 * This header declares the rendering functions for the training analytics
 * interface. These functions draw histograms, graphs, heatmaps, and control
 * panels that provide visual feedback on training progress, network internals,
 * and model performance.
 */

#ifndef GUI_H
#define GUI_H

/**
 * Renders a histogram visualization of weight distribution data.
 *
 * @param x Screen X coordinate of the histogram's top-left corner
 * @param y Screen Y coordinate of the histogram's top-left corner
 * @param w Width of the histogram in pixels
 * @param h Height of the histogram in pixels
 * @param data Pointer to the weight array to visualize
 * @param count Number of elements in the data array
 * @param title Text label displayed above the histogram
 * @param range Maximum absolute value for normalization (values beyond this are
 * clamped)
 */
void DrawHistogram(int x, int y, int w, int h, float *data, int count,
                   const char *title, float range);

/**
 * Renders a dual-axis line graph displaying two time-series datasets.
 *
 * @param x Screen X coordinate of the graph's top-left corner
 * @param y Screen Y coordinate of the graph's top-left corner
 * @param w Width of the graph in pixels
 * @param h Height of the graph in pixels
 * @param d1 Pointer to the first dataset array (loss values)
 * @param d2 Pointer to the second dataset array (accuracy values)
 * @param count Number of data points in each array
 */
void DrawDualGraph(int x, int y, int w, int h, float *d1, float *d2, int count);

/**
 * Renders the live training feed panel showing current batch processing.
 *
 * @param x Screen X coordinate of the panel's top-left corner
 * @param y Screen Y coordinate of the panel's top-left corner
 * @param w Width of the panel in pixels
 * @param h Height of the panel in pixels
 */
void DrawLiveFeed(int x, int y, int w, int h);

/**
 * Renders activation heatmaps for all convolutional layers in the network.
 *
 * @param x Screen X coordinate of the heatmap panel's top-left corner
 * @param y Screen Y coordinate of the heatmap panel's top-left corner
 * @param w Width of the heatmap panel in pixels
 * @param h Height of the heatmap panel in pixels
 */
void DrawLayerHeatmaps(int x, int y, int w, int h);

/**
 * Renders the training statistics and control panel.
 *
 * @param x Screen X coordinate of the panel's top-left corner
 * @param y Screen Y coordinate of the panel's top-left corner
 * @param w Width of the panel in pixels
 * @param h Height of the panel in pixels
 */
void DrawStatsPanel(int x, int y, int w, int h);

#endif
