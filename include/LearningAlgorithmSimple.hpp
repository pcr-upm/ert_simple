/** ****************************************************************************
 *  @file    LearningAlgorithmSimple.hpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2017/06
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef LEARNING_ALGORITHM_SIMPLE_HPP
#define LEARNING_ALGORITHM_SIMPLE_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <LearningAlgorithm.hpp>
#include <RegressionTree.hpp>
#include <TrainingSample.hpp>

namespace upm {

/** ****************************************************************************
 * @class LearningAlgorithmSimple
 * @brief Compute each regression tree in order to complete the cascade.
 ******************************************************************************/
class LearningAlgorithmSimple : public LearningAlgorithm
{
public:
  LearningAlgorithmSimple(impl::SplitType split_type) : LearningAlgorithm(split_type) {};

  ~LearningAlgorithmSimple() {};

  void
  robustEstimation
    (
    const std::vector<cv::Mat> &shapes,
    const std::vector<cv::Mat> &labels,
    cv::Mat &robust_shape,
    cv::Mat &robust_label
    );

  void
  learnRegressor
    (
    std::vector<TrainingSample> &samples,
    EnsembleTrees &forest,
    std::vector<std::vector<int>> &landmark_per_tree,
    const bool coarse_to_fine,
    const cv::Mat &center,
    const cv::Mat &robust_shape
    );

private:
  impl::RegressionTree
  makeRegressionTree
    (
    std::vector<TrainingSample> &samples,
    const std::vector<int> &part_idx
    );

  impl::SplitFeature
  generateBestSplit
    (
    const std::vector<TrainingSample> &samples,
    const std::vector<cv::Mat> &gradients,
    const std::vector<cv::Mat> &Ws,
    const std::vector<int> &part_idx,
    unsigned int split_idx,
    std::vector< std::vector<int> > &node_indices,
    std::vector<cv::Mat> &sum_gradients,
    std::vector<cv::Mat> &sum_Ws
    );

  mutable cv::RNG _rnd = 0xffffffff;
  unsigned int _num_leaf_nodes, _num_split_nodes, _num_nodes;
};

} // namespace upm

#endif /* LEARNING_ALGORITHM_SIMPLE_HPP */
