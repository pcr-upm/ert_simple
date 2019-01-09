/** ****************************************************************************
 *  @file    ChannelFeaturesSimple.cpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2017/06
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ----------------------- INCLUDES --------------------------------------------
#include <ChannelFeaturesSimple.hpp>

namespace upm {

const unsigned int NUM_SAMPLING = 10;

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
ChannelFeaturesSimple::ChannelFeaturesSimple
  (
  const cv::Mat &shape,
  const cv::Mat &label
  )
{
  _robust_shape = shape.clone();
  _robust_label = label.clone();
  float max_diameter = 0.15f;
  cv::RNG rnd = 0xffffffff;
  std::vector<cv::Point2f> random_pattern(NUM_SAMPLING);
  for (cv::Point2f &rnd_point : random_pattern)
    rnd_point = cv::Point2f(rnd.uniform(-1.0f,1.0f), rnd.uniform(-1.0f,1.0f)) * max_diameter;

  /// Generate pixel locations relative to robust shape
  _encoder.reset(new NearestEncoding());
  std::vector<cv::Point2f> pixel_coordinates = _encoder->generatePixelSampling(_robust_shape, random_pattern);
  _encoder->setPixelSamplingEncoding(_robust_shape, _robust_label, pixel_coordinates);
};

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
std::vector<cv::Mat>
ChannelFeaturesSimple::generateChannels
  (
  const cv::Mat &img,
  const cv::Rect_<float> &bbox
  )
{
  // Single gray-scale channel image
  std::vector<cv::Mat> img_channels(1);
  cv::cvtColor(img, img_channels[0], cv::COLOR_BGR2GRAY);
  return img_channels;
};

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
cv::Mat
ChannelFeaturesSimple::extractFeatures
  (
  const std::vector<cv::Mat> &img_channels,
  const float face_height,
  const cv::Mat &rigid,
  const cv::Mat &tform,
  const cv::Mat &shape,
  float level
  )
{
  /// Compute features from a pixel coordinates difference using a pattern
  cv::Mat CR = rigid.colRange(0,2).clone();
  std::vector<cv::Point2f> current_pixel_coordinates = _encoder->getProjectedPixelSampling(CR, tform, shape);

  const unsigned int num_landmarks = static_cast<unsigned int>(shape.rows);
  cv::Mat features = cv::Mat(num_landmarks,NUM_SAMPLING,CV_32FC1);
  for (unsigned int i=0; i < num_landmarks; i++)
    for (unsigned int j=0; j < NUM_SAMPLING; j++)
    {
      int x = static_cast<int>(std::roundf(current_pixel_coordinates[(i*NUM_SAMPLING)+j].x));
      int y = static_cast<int>(std::roundf(current_pixel_coordinates[(i*NUM_SAMPLING)+j].y));
      x = x < 0 ? 0 : x;
      y = y < 0 ? 0 : y;
      x = x > img_channels[0].cols-1 ? img_channels[0].cols-1 : x;
      y = y > img_channels[0].rows-1 ? img_channels[0].rows-1 : y;
      features.at<float>(i,j) = img_channels[0].at<uchar>(y,x);
    }
//  cv::RNG rnd = 0xffffffff;
//  cv::Mat image_gray, image;
//  img_channels[0].convertTo(image_gray, CV_8UC3);
//  cv::cvtColor(image_gray, image, CV_GRAY2BGR);
//  for (unsigned int i=0; i < num_landmarks; i++)
//  {
//    cv::Scalar color = cv::Scalar(rnd.uniform(0,255),rnd.uniform(0,255),rnd.uniform(0,255));
//    for (unsigned int j=0; j < NUM_SAMPLING; j++)
//      cv::circle(image, current_pixel_coordinates[(i*NUM_SAMPLING)+j], 3, color);
//  }
//  std::cout << features.at<float>(0,0) << std::endl;
//  cv::circle(image, current_pixel_coordinates[0], 3, cv::Scalar(255,255,255));
//  cv::imshow("extractFeatures", image);
//  cv::waitKey(0);
  return features;
};

} // namespace upm
