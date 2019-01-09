/** ****************************************************************************
 *  @file    EnsembleTrees.hpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2017/06
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef ENSEMBLE_TREES_HPP
#define ENSEMBLE_TREES_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <utils.hpp>
#include <FaceAnnotation.hpp>
#include <ChannelFeatures.hpp>
#include <ChannelFeaturesSimple.hpp>
#include <LearningAlgorithm.hpp>
#include <transformation.hpp>
#include <cereal/access.hpp>
#include <cereal/types/vector.hpp>
#include <serialization.hpp>
#include <opencv2/opencv.hpp>
#include <numeric>

namespace upm {

/** ****************************************************************************
 * @class EnsembleTrees
 * @brief Use cascade of regression trees that can localize facial landmarks.
 * Predict several update vectors that best agrees with the image data.
 ******************************************************************************/
class EnsembleTrees
{
public:
  /// Train predictor
  EnsembleTrees
    (
    const cv::Size2f shape_size,
    const cv::Mat &robust_shape,
    const cv::Mat &robust_label,
    const std::vector<LearningAlgorithm::EnsembleTrees> &forests
    )
  {
    _shape_size = shape_size;
    _robust_shape = robust_shape.clone();
    _robust_label = robust_label.clone();
    _forests = forests;
    _desc.reset(new ChannelFeaturesSimple(robust_shape, robust_label));
  };

  /// Test predictor
  EnsembleTrees() {};

  ~EnsembleTrees() {};

  void
  process
    (
    const cv::Mat &img,
    const float &scale,
    FaceAnnotation &face,
    const FaceAnnotation &ann
    ) const
  {
    /// Resize bounding box annotation
    FaceBox box_scaled = face.bbox;
    box_scaled.pos.x *= scale;
    box_scaled.pos.y *= scale;
    box_scaled.pos.width  *= scale;
    box_scaled.pos.height *= scale;

    /// Map shape from normalized space into the image dimension
    const unsigned int num_landmarks = _robust_shape.rows;
    cv::Mat utform = unnormalizingTransform(box_scaled.pos, _shape_size);

    /// Load channels into memory only once
    std::vector<cv::Mat> img_channels = _desc->generateChannels(img, box_scaled.pos);

    /// Run algorithm with the robust initialization
    cv::Mat center = cv::repeat((cv::Mat_<float>(1,3) << _shape_size.width*0.5f, _shape_size.height*0.5f, 0.0f), num_landmarks, 1);
    std::vector<cv::Mat> current_shapes(1), current_labels(1);
    current_shapes[0] = _robust_shape.clone();
    current_labels[0] = _robust_label.clone();
//    drawShape(current_shapes[0], current_labels[0], img, box_scaled.pos, utform, cv::Size(200,200));

    for (unsigned int i=0; i < _forests.size(); i++) /// T
    {
      float level = static_cast<float>(i) / static_cast<float>(_forests.size());
      /// Global similarity transform that maps 'robust_shape' to 'current_shape'
      cv::Mat rigid = findSimilarityTransform(_robust_shape, current_shapes[0], current_labels[0]);
      cv::Mat inv;
      cv::invertAffineTransform(rigid, inv);
      cv::Mat features = _desc->extractFeatures(img_channels, box_scaled.pos.height, rigid, utform, current_shapes[0], level);

      /// Compute residuals according to feature values
      std::vector< std::vector<cv::Mat> > residuals(_forests[i].size());
      for (unsigned int j=0; j < _forests[i].size(); j++)
      {
        residuals[j].resize(_forests[i][j].size());
        for (unsigned int k=0; k < _forests[i][j].size(); k++)
          residuals[j][k] = _forests[i][j][k].leafs[_forests[i][j][k].predict(features)].residual;
      }

      /// Update 'current_shape' estimation
//      applyRigidTransformToShape(inv, center, current_shapes[0]);
      for (const std::vector<cv::Mat> &residual : residuals)
        for (const cv::Mat &residual_part : residual)
          addResidualToShape(residual_part, current_shapes[0]);
//      applyRigidTransformToShape(rigid, center, current_shapes[0]);
//      drawShape(current_shapes[0], current_labels[0], img, box_scaled.pos, utform, cv::Size(200,200));
    }
    shapeToFaceParts(current_shapes[0], current_labels[0], utform, scale, face.parts);
  };

  static void
  drawShape
    (
    const cv::Mat &shape,
    const cv::Mat &label,
    const cv::Mat &img,
    const cv::Rect_<float> &bbox,
    const cv::Mat &utform,
    const cv::Size &face_size
    )
  {
    cv::Mat pts, pts_proj;
    for (unsigned int i=0; i < shape.rows; i++)
      pts_proj.push_back(cv::Point2f(shape.at<float>(i,0), shape.at<float>(i,1)));
    cv::transform(pts_proj, pts, utform);
    /// Transform shape to cropped image
    cv::Mat img_T, T = (cv::Mat_<float>(2,3) << 1, 0, -bbox.x, 0, 1, -bbox.y);
    cv::warpAffine(img, img_T, T, img.size());
    cv::Mat img_S, S = (cv::Mat_<float>(2,3) << face_size.width/bbox.width, 0, 0, 0, face_size.height/bbox.height, 0);
    cv::warpAffine(img_T, img_S, S, face_size);
    const cv::Mat ntform = normalizingTransform(bbox, face_size);
    cv::transform(pts, pts_proj, ntform);
    cv::Scalar blue_color(255,0,0), green_color(0,255,0), red_color(0,0,255);
    for (unsigned int i=0; i < pts_proj.rows; i++)
    {
      if (label.at<float>(i) == 1.0f)
        cv::circle(img_S, pts_proj.at<cv::Point2f>(i), 3, shape.at<float>(i,2) > 0.5f ? green_color : blue_color);
      else
        cv::circle(img_S, pts_proj.at<cv::Point2f>(i), 3, red_color);
    }
    cv::imshow("drawShape", img_S);
    cv::waitKey(0);
  };

  friend class cereal::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned version)
  {
    ar & _shape_size & _robust_shape & _robust_label & _forests & _desc;
  };

private:
  cv::Size2f _shape_size;
  cv::Mat _robust_shape;
  cv::Mat _robust_label;
  std::vector<LearningAlgorithm::EnsembleTrees> _forests;
  std::shared_ptr<ChannelFeatures> _desc;
};

} // namespace upm

#endif /* ENSEMBLE_TREES_HPP */
