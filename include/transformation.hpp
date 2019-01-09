/** ****************************************************************************
 *  @file    transformation.hpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2015/06
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef TRANSFORMATION_HPP
#define TRANSFORMATION_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <FaceAnnotation.hpp>
#include <opencv2/opencv.hpp>

namespace upm {

inline cv::Mat
normalizingTransform
  (
  const cv::Rect_<float> &bbox,
  const cv::Size &size
  )
{
  /// Transform that maps bbox.tl() to (0,0) and bbox.br() to (width,height)
  float x = bbox.x, y = bbox.y, width = bbox.width, height = bbox.height;
  return (cv::Mat_<float>(2,3) << size.width/width, 0.0f, (-x*size.width)/width, 0.0f, size.height/height, (-y*size.height)/height);
};

inline cv::Mat
unnormalizingTransform
  (
  const cv::Rect_<float> &bbox,
  const cv::Size &size
  )
{
  /// Transform that maps (0,0) to bbox.tl() and (width,height) to bbox.br()
  float x = bbox.x, y = bbox.y, width = bbox.width, height = bbox.height;
  return (cv::Mat_<float>(2,3) << width/size.width, 0.0f, x, 0.0f, height/size.height, y);
};

inline void
addResidualToShape
  (
  const cv::Mat &residual,
  cv::Mat &shape
  )
{
  /// Sum residual ('x' and 'y' and 'visible')
  cv::Mat pts = shape.colRange(0,residual.cols).clone();
  shape = pts + residual;
};

inline void
applyRigidTransformToShape
  (
  const cv::Mat &rigid,
  const cv::Mat &center,
  cv::Mat &shape
  )
{
  /// Transform points ('x' and 'y') and clone 'visible
  addResidualToShape(-center, shape);
  cv::Mat pts_tformed, pts = shape.colRange(0,2).clone();
  cv::transform(pts.reshape(pts.cols,0), pts_tformed, rigid);
  cv::hconcat(pts_tformed.reshape(pts_tformed.cols,0), shape.col(2), shape);
  addResidualToShape(center, shape);
};

inline void
facePartsToShape
  (
  const std::vector<FacePart> &parts,
  const cv::Mat &tform,
  const float &scale,
  cv::Mat &shape,
  cv::Mat &label
  )
{
  /// Non labelled landmarks are not stored in 'parts' but we need them for 'shape' and 'label'
  std::vector<cv::Point2f> point(1), point_proj(1);
  unsigned int shape_idx = 0;
  for (const std::pair<FacePartLabel,std::vector<int>> &db_part : DB_PARTS)
    for (int feature_idx : db_part.second)
    {
      for (const FacePart &part : parts)
        for (const FaceLandmark &landmark : part.landmarks)
          if (landmark.feature_idx == feature_idx)
          {
            point[0] = landmark.pos * scale;
            cv::transform(point, point_proj, tform);
            shape.at<float>(shape_idx,0) = point_proj[0].x;
            shape.at<float>(shape_idx,1) = point_proj[0].y;
            shape.at<float>(shape_idx,2) = static_cast<float>(landmark.visible);
            label.at<float>(shape_idx,0) = 1.0f;
          }
      shape_idx++;
    }
};

inline void
shapeToFaceParts
  (
  const cv::Mat &shape,
  const cv::Mat &label,
  const cv::Mat &tform,
  const float &scale,
  std::vector<FacePart> &parts
  )
{
  /// Avoid to store non labelled landmarks in 'parts'
  cv::Mat points, points_proj, visible;
  points = shape.colRange(0,2).reshape(2).clone();
  visible = shape.col(2).clone();
  cv::transform(points, points_proj, tform);
  unsigned int shape_idx = 0;
  for (const std::pair<FacePartLabel,std::vector<int>> &db_part : DB_PARTS)
  {
    parts[db_part.first].landmarks.clear();
    for (int feature_idx : db_part.second)
    {
      if (label.at<float>(shape_idx,0) == 1.0f)
      {
        FaceLandmark landmark;
        landmark.feature_idx = feature_idx;
        landmark.pos.x = points_proj.at<float>(shape_idx,0) / scale;
        landmark.pos.y = points_proj.at<float>(shape_idx,1) / scale;
        landmark.visible = visible.at<float>(shape_idx,0) > 0.5f;
        parts[db_part.first].landmarks.push_back(landmark);
      }
      shape_idx++;
    }
  }
};

inline cv::Mat
findSimilarityTransform
  (
  const cv::Mat &src_shape,
  const cv::Mat &dst_shape,
  const cv::Mat &label
  )
{
  /// Similarity matrix which transforms 'mean shape' to 'current shape' (rigid deformation).
  /// Calculating the similarity transform at test time the most computationally expensive
  /// part of this process, is only done once at each level of the cascade.
  cv::Mat src_points, dst_points;
  for (unsigned int i=0; i < src_shape.rows; i++)
  {
    if (label.at<float>(i,0) > 0.5f)
    {
      src_points.push_back(src_shape.row(i).colRange(0,2).reshape(2));
      dst_points.push_back(dst_shape.row(i).colRange(0,2).reshape(2));
    }
  }
//  cv::Mat rigid = cv::estimateRigidTransform(src_points, dst_points, false);
//  if (rigid.empty())
//    rigid = cv::Mat::eye(2,3,CV_64FC1);
//  return rigid;

  cv::Scalar_<float> src_mean = cv::mean(src_points);
  cv::Scalar_<float> dst_mean = cv::mean(dst_points);
  float src_sigma = 0;
  cv::Mat covar = cv::Mat::zeros(2,2,CV_32FC1);
  for (unsigned int i=0; i < src_points.rows; i++)
  {
    cv::Mat src_diff = cv::Mat(src_points.row(i) - src_mean).reshape(1);
    cv::Mat dst_diff = cv::Mat(dst_points.row(i) - dst_mean).reshape(1);
    src_sigma += cv::Mat(src_diff * src_diff.t()).at<float>(0,0);
    covar += cv::Mat(dst_diff.t() * src_diff);
  }
  src_sigma /= static_cast<float>(src_points.rows);
  covar *= (1.0f/static_cast<float>(src_points.rows));

  cv::Mat W, U, V;
  cv::SVD::compute(covar, W, U, V);
  cv::Mat S = cv::Mat::eye(2,2,CV_32FC1);
  if (cv::determinant(covar) < 0)
  {
    if (W.at<float>(1,0) < W.at<float>(0,0))
      S.at<float>(1,1) = -1;
    else
      S.at<float>(0,0) = -1;
  }
  float C = 1.0f;
  if (src_sigma != 0)
    C = (1.0f/src_sigma) * cv::sum(S*W).val[0];
  cv::Mat R = U * S * V;

  cv::Mat CR = C*R;
  cv::Mat rigid, T = cv::Mat(dst_mean).rowRange(0,2) - (CR * cv::Mat(src_mean).rowRange(0,2));
  cv::hconcat(CR, T, rigid);
  return rigid;
};

} // namespace upm

#endif /* TRANSFORMATION_HPP */
