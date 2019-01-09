/** ****************************************************************************
 *  @file    NearestEncoding.cpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2015/06
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ----------------------- INCLUDES --------------------------------------------
#include <NearestEncoding.hpp>

namespace upm {

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
void
NearestEncoding::setPixelSamplingEncoding
  (
  const cv::Mat &shape,
  const cv::Mat &label,
  const std::vector<cv::Point2f> &pixel_coordinates
  )
{
  // Landmarks localization by computing the similarity transform that maps a
  // mean shape to the shape prediction
  const unsigned int num_landmarks = static_cast<unsigned int>(shape.rows);
  const unsigned int num_pixel_coordinates = static_cast<unsigned int>(pixel_coordinates.size());
  _shape_idx.resize(num_pixel_coordinates);
  _deltas.resize(num_pixel_coordinates);
  for (unsigned int i=0; i < num_pixel_coordinates; i++)
  {
    // Find the nearest facial landmark in the shape to this pixel
    float dist, best_dist = std::numeric_limits<float>::max();
    for (unsigned int j=0; j < num_landmarks; j++)
    {
      if (label.at<float>(j,0) > 0.5f)
      {
        dist = static_cast<float>(cv::norm(pixel_coordinates[i] - cv::Point2f(shape.at<float>(j,0),shape.at<float>(j,1))));
        if (dist < best_dist)
        {
          best_dist = dist;
          _shape_idx[i] = j;
        }
      }
    }
    _deltas[i] = pixel_coordinates[i] - cv::Point2f(shape.at<float>(_shape_idx[i],0),shape.at<float>(_shape_idx[i],1));
  }
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
std::vector<cv::Point2f>
NearestEncoding::getProjectedPixelSampling
  (
  const cv::Mat &rigid,
  const cv::Mat &tform,
  const cv::Mat &shape
  )
{
  // Apply global similarity transform that maps 'pixel_coordinates' to 'current_shape'
  std::vector<cv::Point2f> deltas_proj;
  cv::transform(_deltas, deltas_proj, rigid);

  // Gray pixel value relative to current shape in image that corresponds to the
  // pixel identified by shape_idx[i] and deltas[i]
  const unsigned int num_pixel_coordinates = static_cast<unsigned int>(_shape_idx.size());
  std::vector<cv::Point2f> proj_pixel_coordinates(num_pixel_coordinates);
  for (unsigned int i=0; i < num_pixel_coordinates; i++)
  {
    std::vector<cv::Point2f> pt(1), pt_proj(1);
    pt[0] = deltas_proj[i] + cv::Point2f(shape.at<float>(_shape_idx[i],0),shape.at<float>(_shape_idx[i],1));
    cv::transform(pt, pt_proj, tform);
    proj_pixel_coordinates[i] = pt_proj[0];
  }
  return proj_pixel_coordinates;
};

} // namespace upm
