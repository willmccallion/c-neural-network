#ifndef IMAGE_PROC_H
#define IMAGE_PROC_H

void downscale_input(float *high_res, float *low_res);
void center_input(float *src, float *dst);
void apply_brush_high_res(float *grid, int mx, int my);

#endif
