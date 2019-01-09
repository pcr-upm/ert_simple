/** ****************************************************************************
 *  @file    ChannelFeaturesSimple.hpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2017/06
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef CHANNEL_FEATURES_SIMPLE_HPP
#define CHANNEL_FEATURES_SIMPLE_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <ChannelFeatures.hpp>
#include <FeaturesRelativeEncoding.hpp>
#include <NearestEncoding.hpp>
#include <serialization.hpp>
#include <cereal/access.hpp>
#include <cereal/types/polymorphic.hpp>
#include <opencv2/opencv.hpp>

namespace upm {

/** ****************************************************************************
 * @class ChannelFeaturesSimple
 * @brief Extract pixel features from gray-scale channel.
 ******************************************************************************/
class ChannelFeaturesSimple : public ChannelFeatures
{
public:
  ChannelFeaturesSimple() {};

  ChannelFeaturesSimple
    (
    const cv::Mat &shape,
    const cv::Mat &label
    );

  virtual
  ~ChannelFeaturesSimple() {};

  cv::Rect_<float>
  enlargeBbox
    (
    const cv::Rect_<float> &bbox
    ) { return bbox; }

  void
  loadChannelsGenerator() {};

  std::vector<cv::Mat>
  generateChannels
    (
    const cv::Mat &img,
    const cv::Rect_<float> &bbox
    );

  void
  loadFeaturesDescriptor() {};

  cv::Mat
  extractFeatures
    (
    const std::vector<cv::Mat> &img_channels,
    const float face_height,
    const cv::Mat &rigid,
    const cv::Mat &tform,
    const cv::Mat &shape,
    float level
    );

  friend class cereal::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned version)
  {
    ar & _robust_shape & _robust_label & _encoder;
  };

protected:
  cv::Mat _robust_shape, _robust_label;
  std::shared_ptr<FeaturesRelativeEncoding> _encoder;
};

} // namespace upm

CEREAL_REGISTER_TYPE(upm::ChannelFeaturesSimple);
CEREAL_REGISTER_POLYMORPHIC_RELATION(upm::ChannelFeatures, upm::ChannelFeaturesSimple);

#endif /* CHANNEL_FEATURES_SIMPLE_HPP */
