/*
 * Copyright (C) 2019 Kirk Scheper <kirkscheper@gmail.com>
 *
 * This file is part of Paparazzi.
 *
 * Paparazzi is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
 * any later version.
 *
 * Paparazzi is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Paparazzi; see the file COPYING.  If not, write to
 * the Free Software Foundation, 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

/**
 * @file modules/computer_vision/cv_detect_object.h
 * Assumes the object consists of a continuous color and checks
 * if you are over the defined object or not
 */

// Own header
#include "modules/computer_vision/cv_detect_color_object.h"
#include "modules/computer_vision/cv.h"
#include "modules/core/abi.h"
#include "std.h"

#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include "pthread.h"

// vision algo libraries
// #include <sw/ext/opencv_bebop/opencv/modules/core/include/opencv2/core/core_c.h>
// #include <sw/ext/opencv_bebop/opencv/modules/imgproc/include/opencv2/imgproc/imgproc_c.h>
// #include <sw/ext/opencv_bebop/opencv/include/opencv2/opencv.hpp>
// #include <opencv2/core/core.hpp>
// #include <opencv2/imgproc/imgproc.hpp>
// #include <opencv2/highgui/highgui.hpp>
// #include <sys/time.h>
// #include <stdint.h>

// #include <iostream>
// #include <vector>
// #include <dirent.h>
// #include <algorithm>
// #include <string>

#define PRINT(string,...) fprintf(stderr, "[object_detector->%s()] " string,__FUNCTION__ , ##__VA_ARGS__)
#define OBJECT_DETECTOR_VERBOSE TRUE

#if OBJECT_DETECTOR_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

static pthread_mutex_t mutex;

#ifndef COLOR_OBJECT_DETECTOR_FPS1
#define COLOR_OBJECT_DETECTOR_FPS1 0 ///< Default FPS (zero means run at camera fps)
#endif
#ifndef COLOR_OBJECT_DETECTOR_FPS2
#define COLOR_OBJECT_DETECTOR_FPS2 0 ///< Default FPS (zero means run at camera fps)
#endif
#ifndef COLOR_OBJECT_DETECTOR_FPS3
#define COLOR_OBJECT_DETECTOR_FPS3 0 ///< Default FPS (zero means run at camera fps)
#endif

#define NUM_STRIPS 5 // 5 vertical strips

// Filter Settings_
uint8_t cod_lum_min1 = 0;
uint8_t cod_lum_max1 = 0;
uint8_t cod_cb_min1 = 0;
uint8_t cod_cb_max1 = 0;
uint8_t cod_cr_min1 = 0;
uint8_t cod_cr_max1 = 0;

uint8_t cod_lum_min2 = 0;
uint8_t cod_lum_max2 = 0;
uint8_t cod_cb_min2 = 0;
uint8_t cod_cb_max2 = 0;
uint8_t cod_cr_min2 = 0;
uint8_t cod_cr_max2 = 0;

//Im assuming this is needed for the black filter
uint8_t cod_lum_min3 = 0;
uint8_t cod_lum_max3 = 0;
uint8_t cod_cb_min3 = 0;
uint8_t cod_cb_max3 = 0;
uint8_t cod_cr_min3 = 0;
uint8_t cod_cr_max3 = 0;
uint32_t pixel_array[NUM_STRIPS] = {};

bool cod_draw1 = false;
bool cod_draw2 = false;
//Im assuming this is needed for the black filter
bool cod_draw3 = false;

// define global variables
struct color_object_t {
  int32_t x_c;
  int32_t y_c;
  uint32_t color_count[NUM_STRIPS];
  // uint32_t left_orange;
  // uint32_t right_orange;
  // uint32_t left_green;
  // uint32_t right_green;
  // uint32_t left_black;
  // uint32_t right_black;
  uint32_t heading_idx;
  bool updated;
};
struct color_object_t global_filters[3]; //changed from 2 to 3 to take the black filter into account

// Function
uint32_t* find_object_centroid(struct image_t *img, int32_t* p_xc, int32_t* p_yc, bool draw,
                              uint8_t lum_min, uint8_t lum_max,
                              uint8_t cb_min, uint8_t cb_max,
                              uint8_t cr_min, uint8_t cr_max,  
                              uint32_t* heading_idx, uint32_t* pixel_array);

/*
 * object_detector
 * @param img - input image to process
 * @param filter - which detection filter to process
 * @return img
 */
static struct image_t *object_detector(struct image_t *img, uint8_t filter)
{
  uint8_t lum_min, lum_max;
  uint8_t cb_min, cb_max;
  uint8_t cr_min, cr_max;
  bool draw;

  switch (filter){
    case 1:
      lum_min = cod_lum_min1;
      lum_max = cod_lum_max1;
      cb_min = cod_cb_min1;
      cb_max = cod_cb_max1;
      cr_min = cod_cr_min1;
      cr_max = cod_cr_max1;
      draw = cod_draw1;
      break;
    case 2:
      lum_min = cod_lum_min2;
      lum_max = cod_lum_max2;
      cb_min = cod_cb_min2;
      cb_max = cod_cb_max2;
      cr_min = cod_cr_min2;
      cr_max = cod_cr_max2;
      draw = cod_draw2;
      break;
    //Im assuming this is needed for black
    case 3:
      lum_min = cod_lum_min3;
      lum_max = cod_lum_max3;
      cb_min = cod_cb_min3;
      cb_max = cod_cb_max1;
      cr_min = cod_cr_min3;
      cr_max = cod_cr_max3;
      draw = cod_draw3;
      break;
    default:
      return img;
  };

  int32_t x_c, y_c;
  // uint32_t left_orange = 0;
  // uint32_t right_orange = 0;
  // uint32_t left_green = 0;
  // uint32_t right_green = 0;
  // uint32_t left_black = 0;
  // uint32_t right_black = 0;
  uint32_t heading_idx = 0;

  // Filter and find centroid
  uint32_t count = find_object_centroid(img, &x_c, &y_c, draw, lum_min, lum_max, cb_min, cb_max, cr_min, cr_max, &heading_idx, pixel_array);
  //VERBOSE_PRINT("Color count %d: %u, threshold %u, x_c %d, y_c %d\n", camera, object_count, count_threshold, x_c, y_c);
  //VERBOSE_PRINT("centroid %d: (%d, %d) r: %4.2f a: %4.2f\n", camera, x_c, y_c,
  //     hypotf(x_c, y_c) / hypotf(img->w * 0.5, img->h * 0.5), RadOfDeg(atan2f(y_c, x_c)));

  pthread_mutex_lock(&mutex);
  memcpy(global_filters[filter-1].color_count, count, sizeof(NUM_STRIPS));
  global_filters[filter-1].heading_idx = heading_idx;

  global_filters[filter-1].x_c = x_c;
  global_filters[filter-1].y_c = y_c;
  global_filters[filter-1].updated = true;
  pthread_mutex_unlock(&mutex);

  return img;
}

struct image_t *object_detector1(struct image_t *img, uint8_t camera_id);
struct image_t *object_detector1(struct image_t *img, uint8_t camera_id __attribute__((unused)))
{
  return object_detector(img, 1);
}

struct image_t *object_detector2(struct image_t *img, uint8_t camera_id);
struct image_t *object_detector2(struct image_t *img, uint8_t camera_id __attribute__((unused)))
{
  return object_detector(img, 2);
}
// Im assuming this is for black filter
struct image_t *object_detector3(struct image_t *img, uint8_t camera_id);
struct image_t *object_detector3(struct image_t *img, uint8_t camera_id __attribute__((unused)))
{
  return object_detector(img, 3);
}

void color_object_detector_init(void)
{
  memset(global_filters, 0, 3*sizeof(struct color_object_t)); 
  pthread_mutex_init(&mutex, NULL);

// ORANGE
#ifdef COLOR_OBJECT_DETECTOR_CAMERA1
#ifdef COLOR_OBJECT_DETECTOR_LUM_MIN1
  cod_lum_min1 = COLOR_OBJECT_DETECTOR_LUM_MIN1;
  cod_lum_max1 = COLOR_OBJECT_DETECTOR_LUM_MAX1;
  cod_cb_min1 = COLOR_OBJECT_DETECTOR_CB_MIN1;
  cod_cb_max1 = COLOR_OBJECT_DETECTOR_CB_MAX1;
  cod_cr_min1 = COLOR_OBJECT_DETECTOR_CR_MIN1;
  cod_cr_max1 = COLOR_OBJECT_DETECTOR_CR_MAX1;
#endif
#ifdef COLOR_OBJECT_DETECTOR_DRAW1
  cod_draw1 = COLOR_OBJECT_DETECTOR_DRAW1;
#endif

  cv_add_to_device(&COLOR_OBJECT_DETECTOR_CAMERA1, object_detector1, COLOR_OBJECT_DETECTOR_FPS1, 0);
#endif

// GREEN
#ifdef COLOR_OBJECT_DETECTOR_CAMERA2
#ifdef COLOR_OBJECT_DETECTOR_LUM_MIN2
  cod_lum_min2 = COLOR_OBJECT_DETECTOR_LUM_MIN2;
  cod_lum_max2 = COLOR_OBJECT_DETECTOR_LUM_MAX2;
  cod_cb_min2 = COLOR_OBJECT_DETECTOR_CB_MIN2;
  cod_cb_max2 = COLOR_OBJECT_DETECTOR_CB_MAX2;
  cod_cr_min2 = COLOR_OBJECT_DETECTOR_CR_MIN2;
  cod_cr_max2 = COLOR_OBJECT_DETECTOR_CR_MAX2;
#endif
#ifdef COLOR_OBJECT_DETECTOR_DRAW2
  cod_draw2 = COLOR_OBJECT_DETECTOR_DRAW2;
#endif
  cv_add_to_device(&COLOR_OBJECT_DETECTOR_CAMERA2, object_detector2, COLOR_OBJECT_DETECTOR_FPS2, 1);
#endif

// BLACK
#ifdef COLOR_OBJECT_DETECTOR_CAMERA3
#ifdef COLOR_OBJECT_DETECTOR_LUM_MIN3
  cod_lum_min3 = COLOR_OBJECT_DETECTOR_LUM_MIN3;
  cod_lum_max3 = COLOR_OBJECT_DETECTOR_LUM_MAX3;
  cod_cb_min3 = COLOR_OBJECT_DETECTOR_CB_MIN3;
  cod_cb_max3 = COLOR_OBJECT_DETECTOR_CB_MAX3;
  cod_cr_min3 = COLOR_OBJECT_DETECTOR_CR_MIN3;
  cod_cr_max3 = COLOR_OBJECT_DETECTOR_CR_MAX3;
#endif
#ifdef COLOR_OBJECT_DETECTOR_DRAW3
  cod_draw3 = COLOR_OBJECT_DETECTOR_DRAW3;
#endif
  cv_add_to_device(&COLOR_OBJECT_DETECTOR_CAMERA3, object_detector3, COLOR_OBJECT_DETECTOR_FPS3, 2);
#endif
}

/*
 * find_object_centroid
 *
 * Finds the centroid of pixels in an image within filter bounds.
 * Also returns the amount of pixels that satisfy these filter bounds.
 *
 * @param img - input image to process formatted as YUV422.
 * @param p_xc - x coordinate of the centroid of color object
 * @param p_yc - y coordinate of the centroid of color object
 * @param lum_min - minimum y value for the filter in YCbCr colorspace
 * @param lum_max - maximum y value for the filter in YCbCr colorspace
 * @param cb_min - minimum cb value for the filter in YCbCr colorspace
 * @param cb_max - maximum cb value for the filter in YCbCr colorspace
 * @param cr_min - minimum cr value for the filter in YCbCr colorspace
 * @param cr_max - maximum cr value for the filter in YCbCr colorspace
 * @param draw - whether or not to draw on image
 * @return number of pixels of image within the filter bounds.
 */



uint32_t* find_object_centroid(struct image_t *img, int32_t* p_xc, int32_t* p_yc, bool draw,
                              uint8_t lum_min, uint8_t lum_max,
                              uint8_t cb_min, uint8_t cb_max,
                              uint8_t cr_min, uint8_t cr_max,
                              uint32_t* heading_idx, uint32_t* pixel_array
                              ) {
    uint32_t cnt = 0;
    uint32_t tot_x = 0;
    uint32_t tot_y = 0;

    uint8_t *buffer = img->buf;
    uint16_t img_w = img->w;
    uint16_t img_h = img->h;
    uint16_t strip_width = img_w / NUM_STRIPS;

    // Iterate over each pixel
    for (uint16_t y = 0; y < img_h; ++y) {
        for (uint16_t x = 0; x < img_w; ++x) { //from x goes from bottom to top (0 - img_w)
            // Calculate strip index
            uint16_t strip_idx = x / strip_width;

            // Check if the color is inside the specified values
            uint8_t *yp, *up, *vp;
            if (x % 2 == 0) {
                // Even x
                up = &buffer[y * 2 * img_w + 2 * x];      // U
                yp = &buffer[y * 2 * img_w + 2 * x + 1];  // Y1
                vp = &buffer[y * 2 * img_w + 2 * x + 2];  // V
            } else {
                // Uneven x
                up = &buffer[y * 2 * img_w + 2 * x - 2];  // U
                vp = &buffer[y * 2 * img_w + 2 * x];      // V
                yp = &buffer[y * 2 * img_w + 2 * x + 1];  // Y2
            }

            if ((*yp >= lum_min) && (*yp <= lum_max) &&
                (*up >= cb_min) && (*up <= cb_max) &&
                (*vp >= cr_min) && (*vp <= cr_max)) {
                cnt++;

                // Increment pixel count for this strip
                pixel_array[strip_idx]++;

                tot_x += x;
                tot_y += y;
                if (draw) {
                    //*yp = 255;  // make pixel brighter in image
                }
            }
        }
    }


    // Assign the temporary counts to the output variables
    // *left_orange_cnt = pixel_counts[0];
    // *right_orange_cnt = pixel_counts[NUM_STRIPS - 1];
    // *left_green_cnt = green_counts[0];
    // *right_green_cnt = green_counts[NUM_STRIPS - 1];
    // *left_black_cnt = black_counts[0];
    // *right_black_cnt = black_counts[NUM_STRIPS - 1];

    // Calculate centroid
    if (cnt > 0) {
        *p_xc = (int32_t)roundf(tot_x / ((float)cnt) - img_w * 0.5f);
        *p_yc = (int32_t)roundf(img_h * 0.5f - tot_y / ((float)cnt));
    } else {
        *p_xc = 0;
        *p_yc = 0;
    }

    ////////////////////////////
  
    
    return pixel_array;
}

void color_object_detector_periodic(void)
{
  static struct color_object_t local_filters[3];
  pthread_mutex_lock(&mutex);
  memcpy(local_filters, global_filters, 3*sizeof(struct color_object_t));
  pthread_mutex_unlock(&mutex);

  if(local_filters[0].updated){
    AbiSendMsgVISUAL_DETECTION(COLOR_OBJECT_DETECTION1_ID, local_filters[0].x_c, local_filters[0].y_c,
        0, 0, local_filters[0].color_count, local_filters[0].heading_idx, 0);
    local_filters[0].updated = false;
  }
  if(local_filters[1].updated){
    AbiSendMsgVISUAL_DETECTION(COLOR_OBJECT_DETECTION2_ID, local_filters[1].x_c, local_filters[1].y_c,
        0, 0, local_filters[1].color_count, local_filters[0].heading_idx, 1);
    local_filters[1].updated = false;
  }
  // Process black filter
  if(local_filters[2].updated){
    AbiSendMsgVISUAL_DETECTION(COLOR_OBJECT_DETECTION3_ID, local_filters[2].x_c, local_filters[2].y_c,
        0, 0, local_filters[2].color_count, local_filters[2].heading_idx, 2);
    local_filters[2].updated = false;
  }
};
