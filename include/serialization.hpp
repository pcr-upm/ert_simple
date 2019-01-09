/** ****************************************************************************
 *  @file    serialization.hpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2015/06
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef SERIALIZATION_HPP
#define SERIALIZATION_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <opencv2/opencv.hpp>
#include <cereal/archives/binary.hpp>

namespace cereal {

  template<class Archive>
  void serialize(Archive &ar, cv::Point2f &p, const unsigned version)
  {
    ar & p.x & p.y;
  }

  template<class Archive>
  void serialize(Archive &ar, cv::Size2f &s, const unsigned version)
  {
    ar & s.width & s.height;
  }

  template<class Archive>
  void serialize(Archive &ar, cv::Vec3i &v, const unsigned version)
  {
    ar & v.val[0] & v.val[1] & v.val[2];
  }

  template<class Archive>
  void serialize(Archive &ar, cv::Vec2f &v, const unsigned version)
  {
    ar & v.val[0] & v.val[1];
  }

  template<class Archive>
  void save(Archive& ar, const cv::Mat& mat)
  {
    int rows, cols, type;
    bool continuous;

    rows = mat.rows;
    cols = mat.cols;
    type = mat.type();
    continuous = mat.isContinuous();

    ar & rows & cols & type & continuous;

    if (continuous)
    {
      const int data_size = rows * cols * static_cast<int>(mat.elemSize());
      auto mat_data = cereal::binary_data(mat.ptr(), data_size);
      ar & mat_data;
    }
    else {
      const int row_size = cols * static_cast<int>(mat.elemSize());
      for (int i = 0; i < rows; i++)
      {
        auto row_data = cereal::binary_data(mat.ptr(i), row_size);
        ar & row_data;
      }
    }
  };

  template<class Archive>
  void load(Archive& ar, cv::Mat& mat)
  {
    int rows, cols, type;
    bool continuous;

    ar & rows & cols & type & continuous;

    if (continuous)
    {
      mat.create(rows, cols, type);
      const int data_size = rows * cols * static_cast<int>(mat.elemSize());
      auto mat_data = cereal::binary_data(mat.ptr(), data_size);
      ar & mat_data;
    }
    else {
      mat.create(rows, cols, type);
      const int row_size = cols * static_cast<int>(mat.elemSize());
      for (int i = 0; i < rows; i++)
      {
        auto row_data = cereal::binary_data(mat.ptr(i), row_size);
        ar & row_data;
      }
    }
  };

} // namespace cereal

#endif /* SERIALIZATION_HPP */
