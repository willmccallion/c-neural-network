#ifndef GUI_H
#define GUI_H

void DrawHistogram(int x, int y, int w, int h, float *data, int count,
                   const char *title, float range);
void DrawDualGraph(int x, int y, int w, int h, float *d1, float *d2, int count);
void DrawLiveFeed(int x, int y, int w, int h);
void DrawLayerHeatmaps(int x, int y, int w, int h);
void DrawStatsPanel(int x, int y, int w, int h);

#endif
