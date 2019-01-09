/** ****************************************************************************
 *  @file    TrainingSample.hpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2015/06
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef TRAINING_SAMPLE_HPP
#define TRAINING_SAMPLE_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <opencv2/opencv.hpp>

namespace upm {

struct TrainingSample
{
  unsigned int sample_idx;
  unsigned int image_idx;
  cv::Rect_<float> bbox;
  float error;
  cv::Mat target_shape;
  cv::Mat target_label;
  cv::Mat current_shape;
  cv::Mat current_label;
  cv::Mat current_rigid;
  cv::Mat features;

  TrainingSample&
  operator=(const TrainingSample &src)
  {
    sample_idx = src.sample_idx;
    image_idx = src.image_idx;
    bbox = src.bbox;
    error = src.error;
    target_shape = src.target_shape.clone();
    target_label = src.target_label.clone();
    current_shape = src.current_shape.clone();
    current_label = src.current_label.clone();
    current_rigid = src.current_rigid.clone();
    features = src.features.clone();
    return *this;
  }

  bool
  operator<(const TrainingSample &src) const
  {
    return error < src.error;
  }
};

} // namespace upm

#endif /* TRAINING_SAMPLE_HPP */
