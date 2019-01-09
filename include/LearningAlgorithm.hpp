/** ****************************************************************************
 *  @file    LearningAlgorithm.hpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2015/06
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef LEARN_ALGORITHM_HPP
#define LEARN_ALGORITHM_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <RegressionTree.hpp>
#include <TrainingSample.hpp>
#include <utils.hpp>

namespace upm {

/** ****************************************************************************
 * @class LearningAlgorithm
 * @brief Compute each regression tree in order to complete the cascade.
 ******************************************************************************/
class LearningAlgorithm
{
public:
  typedef std::vector< std::vector<impl::RegressionTree> > EnsembleTrees;

  LearningAlgorithm(impl::SplitType split_type) : _split_type(split_type) {};

  virtual
  ~LearningAlgorithm() {};

  virtual void
  robustEstimation
    (
    const std::vector<cv::Mat> &shapes,
    const std::vector<cv::Mat> &labels,
    cv::Mat &robust_shape,
    cv::Mat &robust_label
    ) = 0;

  virtual void
  learnRegressor
    (
    std::vector<TrainingSample> &samples,
    EnsembleTrees &forest,
    std::vector<std::vector<int>> &landmark_per_tree,
    const bool coarse_to_fine,
    const cv::Mat &center,
    const cv::Mat &robust_shape
    ) = 0;

protected:
  impl::SplitType _split_type;
};

} // namespace upm

#endif /* LEARN_ALGORITHM_HPP */
