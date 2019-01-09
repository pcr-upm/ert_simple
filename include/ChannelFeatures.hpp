/** ****************************************************************************
 *  @file    ChannelFeatures.hpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2015/06
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef CHANNEL_FEATURES_HPP
#define CHANNEL_FEATURES_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <cereal/access.hpp>
#include <cereal/types/polymorphic.hpp>
#include <opencv2/opencv.hpp>

namespace upm {

/** ****************************************************************************
 * @class ChannelFeatures
 * @brief Extract features from several channels and projected pixels.
 ******************************************************************************/
class ChannelFeatures
{
public:
  ChannelFeatures() {};

  virtual
  ~ChannelFeatures() {};

  virtual cv::Rect_<float>
  enlargeBbox
    (
    const cv::Rect_<float> &bbox
    ) = 0;

  virtual void
  loadChannelsGenerator() = 0;

  virtual std::vector<cv::Mat>
  generateChannels
    (
    const cv::Mat &img,
    const cv::Rect_<float> &bbox
    ) = 0;

  virtual void
  loadFeaturesDescriptor() = 0;

  virtual cv::Mat
  extractFeatures
    (
    const std::vector<cv::Mat> &img_channels,
    const float face_height,
    const cv::Mat &rigid,
    const cv::Mat &tform,
    const cv::Mat &shape,
    float level
    ) = 0;

  friend class cereal::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned version) {};
};

} // namespace upm

#endif /* CHANNEL_FEATURES_HPP */
