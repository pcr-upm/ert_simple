/** ****************************************************************************
 *  @file    EnsembleTrainer.hpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2017/06
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef ENSEMBLE_TRAINER_HPP
#define ENSEMBLE_TRAINER_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <FaceAnnotation.hpp>
#include <TrainingSample.hpp>
#include <EnsembleTrees.hpp>
#include <opencv2/opencv.hpp>

namespace upm {

/** ****************************************************************************
 * @class EnsembleTrainer
 * @brief Train cascade of regression trees using the gradient boosting
 * algorithm with a sum of square error loss.
 ******************************************************************************/
class EnsembleTrainer
{
public:
  EnsembleTrainer() {};

  ~EnsembleTrainer() {};

  EnsembleTrees
  train
    (
    const std::vector<FaceAnnotation> &anns,
    const std::vector<cv::Mat> &imgs,
    const std::vector<float> scales,
    const ErrorMeasure &measure
    ) const;

private:
  void
  computeError
    (
    const std::vector<FaceAnnotation> &anns,
    const std::vector<cv::Mat> &tform,
    const std::vector<float> &scales,
    const ErrorMeasure &measure,
    std::vector<TrainingSample> &samples,
    float &train_mean,
    float &train_variance
    ) const;

  mutable cv::RNG _rnd = 0xffffffff;
};

} // namespace upm

#endif /* ENSEMBLE_TRAINER_HPP */
