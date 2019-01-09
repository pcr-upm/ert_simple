/** ****************************************************************************
 *  @file    FeaturesRelativeEncoding.hpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2015/06
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef FEATURES_RELATIVE_ENCODING_HPP
#define FEATURES_RELATIVE_ENCODING_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <cereal/access.hpp>
#include <cereal/types/polymorphic.hpp>
#include <opencv2/opencv.hpp>

namespace upm {

/** ****************************************************************************
 * @class FeaturesRelativeEncoding
 * @brief Pixel features extraction related to the mean shape estimation.
 ******************************************************************************/
class FeaturesRelativeEncoding
{
public:
  FeaturesRelativeEncoding() {};

  virtual
  ~FeaturesRelativeEncoding() {};

  std::vector<cv::Point2f>
  generatePixelSampling
    (
    const cv::Mat &shape,
    std::vector<cv::Point2f> sampling_pattern
    )
  {
    /// Generate sampling pattern around landmarks
    const unsigned int num_landmarks = shape.rows;
    const unsigned int num_sampling = sampling_pattern.size();
    std::vector<cv::Point2f> pixel_coordinates(num_landmarks*num_sampling);
    for (unsigned int i=0; i < num_landmarks; i++)
    {
      cv::Point2f landmark = cv::Point2f(shape.at<float>(i,0), shape.at<float>(i,1));
      for (unsigned int j=0; j < num_sampling; j++)
        pixel_coordinates[i*num_sampling+j] = landmark + sampling_pattern[j];
    }
    return pixel_coordinates;
  };

  virtual void
  setPixelSamplingEncoding
    (
    const cv::Mat &shape,
    const cv::Mat &label,
    const std::vector<cv::Point2f> &pixel_coordinates
    ) = 0;

  virtual std::vector<cv::Point2f>
  getProjectedPixelSampling
    (
    const cv::Mat &rigid,
    const cv::Mat &tform,
    const cv::Mat &shape
    ) = 0;

  friend class cereal::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned version) {};
};

} // namespace upm

#endif /* FEATURES_RELATIVE_ENCODING_HPP */
