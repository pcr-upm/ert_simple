/** ****************************************************************************
 *  @file    RegressionTree.hpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2015/06
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef REGRESSION_TREE_HPP
#define REGRESSION_TREE_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <cereal/access.hpp>
#include <serialization.hpp>
#include <opencv2/opencv.hpp>

namespace upm {

namespace impl {

enum SplitType { SINGLE, PAIR };

struct SplitFeature
{
  SplitType type;
  unsigned int lnd;
  unsigned int desc1;
  unsigned int desc2;
  float thresh;

  SplitFeature&
  operator=(SplitFeature src)
  {
    type = src.type;
    lnd = src.lnd;
    desc1 = src.desc1;
    desc2 = src.desc2;
    thresh = src.thresh;
    return *this;
  }

  float
  getDescriptor(const cv::Mat &features) const
  {
    switch (type)
    {
      case SplitType::PAIR:
        return features.at<float>(lnd,desc1) - features.at<float>(lnd,desc2);
      default:
        return features.at<float>(lnd,desc1);
    }
  }

  friend class cereal::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned version)
  {
    ar & type & lnd & desc1 & desc2 & thresh;
  };
};

struct LeafFeature
{
  cv::Mat residual;

  LeafFeature&
  operator=(LeafFeature src)
  {
    residual = src.residual.clone();
    return *this;
  }

  friend class cereal::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned version)
  {
    ar & residual;
  };
};

struct RegressionTree
{
  std::vector<SplitFeature> splits;
  std::vector<LeafFeature> leafs;

  RegressionTree&
  operator=(RegressionTree src)
  {
    splits = src.splits;
    leafs = src.leafs;
    return *this;
  }

  inline unsigned int
  predict
    (
    const cv::Mat &features
    ) const
  {
    unsigned int i = 0;
    const unsigned int num_split_nodes = splits.size();
    while (i < num_split_nodes) // split_nodes
    {
      if (splits[i].getDescriptor(features) > splits[i].thresh)
        i = (2*i)+1; // left child
      else
        i = (2*i)+2; // right child
    }
    return i-num_split_nodes;
  };

  friend class cereal::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned version)
  {
    ar & splits & leafs;
  };
};

} // namespace impl

} // namespace upm

#endif /* REGRESSION_TREE_HPP */
