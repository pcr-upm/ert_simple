/** ****************************************************************************
 *  @file    NearestEncoding.hpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2015/06
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef NEAREST_ENCODING_HPP
#define NEAREST_ENCODING_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <FeaturesRelativeEncoding.hpp>
#include <serialization.hpp>
#include <cereal/access.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/polymorphic.hpp>
#include <opencv2/opencv.hpp>

namespace upm {

/** ****************************************************************************
 * @class NearestEncoding
 * @brief Pixel features extraction related to the mean shape estimation.
 ******************************************************************************/
class NearestEncoding : public FeaturesRelativeEncoding
{
public:
  NearestEncoding() {};

  ~NearestEncoding() {};

  void
  setPixelSamplingEncoding
    (
    const cv::Mat &shape,
    const cv::Mat &label,
    const std::vector<cv::Point2f> &pixel_coordinates
    );

  std::vector<cv::Point2f>
  getProjectedPixelSampling
    (
    const cv::Mat &rigid,
    const cv::Mat &tform,
    const cv::Mat &shape
    );

  friend class cereal::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned version)
  {
    ar & _shape_idx & _deltas;
  };

private:
  std::vector<unsigned int> _shape_idx;
  std::vector<cv::Point2f> _deltas;
};

} // namespace upm

CEREAL_REGISTER_TYPE(upm::NearestEncoding);
CEREAL_REGISTER_POLYMORPHIC_RELATION(upm::FeaturesRelativeEncoding, upm::NearestEncoding);

#endif /* NEAREST_ENCODING_HPP */
