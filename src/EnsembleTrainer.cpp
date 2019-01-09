/** ****************************************************************************
 *  @file    EnsembleTrainer.cpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2017/06
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ----------------------- INCLUDES --------------------------------------------
#include <EnsembleTrainer.hpp>
#include <trace.hpp>
#include <ChannelFeatures.hpp>
#include <ChannelFeaturesSimple.hpp>
#include <LearningAlgorithmSimple.hpp>
#include <transformation.hpp>
#include <numeric>
#include <omp.h>

namespace upm {

const unsigned int NUM_LEVELS = 10;
const int OVERSAMPLING_AMOUNT = 20;
const cv::Size2f SHAPE_SIZE = cv::Size2f(1.0f,1.0f);

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
EnsembleTrees
EnsembleTrainer::train
  (
  const std::vector<FaceAnnotation> &anns,
  const std::vector<cv::Mat> &imgs,
  const std::vector<float> scales,
  const ErrorMeasure &measure
  ) const
{
  /// Resize bounding boxes annotations
  const unsigned int num_data = static_cast<int>(imgs.size());
  std::vector<FaceBox> boxes_scaled(num_data);
  for (unsigned int i=0; i < num_data; i++)
  {
    boxes_scaled[i] = anns[i].bbox;
    boxes_scaled[i].pos.x *= scales[i];
    boxes_scaled[i].pos.y *= scales[i];
    boxes_scaled[i].pos.width  *= scales[i];
    boxes_scaled[i].pos.height *= scales[i];
  }

  /// Map shape from image dimension into the normalized space
  unsigned int num_landmarks = static_cast<int>(DB_LANDMARKS.size());
  std::vector<cv::Mat> shapes(num_data), labels(num_data), utform(num_data);
  for (unsigned int i=0; i < num_data; i++)
  {
    const cv::Mat ntform = normalizingTransform(boxes_scaled[i].pos, SHAPE_SIZE);
    shapes[i] = cv::Mat::zeros(num_landmarks,3,CV_32FC1);
    labels[i] = cv::Mat::zeros(num_landmarks,1,CV_32FC1);
    facePartsToShape(anns[i].parts, ntform, scales[i], shapes[i], labels[i]);
    utform[i] = unnormalizingTransform(boxes_scaled[i].pos, SHAPE_SIZE);
  }

  boost::shared_ptr<LearningAlgorithm> trainer(new LearningAlgorithmSimple(impl::SplitType::PAIR));
  /// Compute robust shape estimation
  cv::Mat robust_shape = cv::Mat::zeros(num_landmarks,3,CV_32FC1);
  cv::Mat robust_label = cv::Mat::zeros(num_landmarks,1,CV_32FC1);
  trainer->robustEstimation(shapes, labels, robust_shape, robust_label);

  boost::shared_ptr<ChannelFeatures> desc(new ChannelFeaturesSimple(robust_shape, robust_label));
  /// Load channels into memory only once
  std::vector< std::vector<cv::Mat> > imgs_channels(num_data);
  for (unsigned int i=0; i < num_data; i++)
    imgs_channels[i] = desc->generateChannels(imgs[i], boxes_scaled[i].pos);

  /// Generate training samples with R initializations for each data
  const unsigned int num_samples = num_data*OVERSAMPLING_AMOUNT;
  UPM_PRINT("Total number of samples: " << num_samples);
  std::vector<TrainingSample> samples(num_samples);
  cv::Mat center = cv::repeat((cv::Mat_<float>(1,3) << SHAPE_SIZE.width*0.5f, SHAPE_SIZE.height*0.5f, 0.0f), num_landmarks, 1);
  for (unsigned int i=0; i < num_data; i++)
  {
    TrainingSample sample;
    sample.image_idx = i;
    sample.bbox = boxes_scaled[i].pos;
    sample.target_shape = shapes[i].clone();
    sample.target_label = labels[i].clone();
    std::vector<int> rnd_indices;
    cv::Mat mask;
    for (unsigned int j=0; j < OVERSAMPLING_AMOUNT; j++)
    {
      sample.sample_idx = (i*OVERSAMPLING_AMOUNT)+j;
      if (j==0)
      {
        /// The initial shapes are what we really use during testing
        sample.current_shape = robust_shape.clone();
        sample.current_label = robust_label.clone();
      }
      else
      {
        /// Pick other random target shape and use that as initial shape
        int rnd_idx;
        do
          rnd_idx = _rnd.uniform(0,num_data);
        while ((i == rnd_idx) or (std::find(rnd_indices.begin(),rnd_indices.end(),rnd_idx) != rnd_indices.end()));
        rnd_indices.push_back(rnd_idx);
        sample.current_shape = robust_shape.clone();
        labels[rnd_idx].convertTo(mask, CV_8U, 255);
        shapes[rnd_idx].copyTo(sample.current_shape, cv::repeat(mask,1,robust_shape.cols));
        sample.current_label = robust_label.clone();
      }
      samples[sample.sample_idx] = sample;
//      EnsembleTrees::drawShape(sample.current_shape, sample.current_label, imgs[i], boxes_scaled[i].pos, utform[i], cv::Size(200,200));
    }
  }

  /// Compute initial training samples error
  float train_mean, train_variance;
  computeError(anns, utform, scales, measure, samples, train_mean, train_variance);
  UPM_PRINT("[-] Training: " << train_mean << " (±" << train_variance << ")");

  /// Create cascade of T [strong regressors 'r'] with K [weak regressors 'g']
  std::vector<LearningAlgorithm::EnsembleTrees> forests(NUM_LEVELS);
  std::vector<float> train_means(NUM_LEVELS), train_variances(NUM_LEVELS);
  for (unsigned int i=0; i < NUM_LEVELS; i++) /// T
  {
    float level = static_cast<float>(i) / static_cast<float>(NUM_LEVELS);
    UPM_PRINT("Extract features...");
    /// Global similarity transform that maps 'robust_shape' to 'current_shape'
    std::vector<cv::Mat> rigids(num_samples), invs(num_samples);
    #pragma omp parallel for
    for (unsigned int j=0; j < num_samples; j++)
    {
      rigids[samples[j].sample_idx] = findSimilarityTransform(robust_shape, samples[j].current_shape, samples[j].current_label);
      cv::invertAffineTransform(rigids[samples[j].sample_idx], invs[samples[j].sample_idx]);
      samples[j].features = desc->extractFeatures(imgs_channels[samples[j].image_idx], boxes_scaled[samples[j].image_idx].pos.height, rigids[samples[j].sample_idx], utform[samples[j].image_idx], samples[j].current_shape, level);
    }
//    /// Apply similarity transform
//    for (TrainingSample &sample : samples)
//    {
//      applyRigidTransformToShape(invs[sample.sample_idx], center, sample.current_shape);
//      applyRigidTransformToShape(invs[sample.sample_idx], center, sample.target_shape);
//    }
    UPM_PRINT("Fitting trees...");
    std::vector<std::vector<int>> landmark_per_tree;
    trainer->learnRegressor(samples, forests[i], landmark_per_tree, false, center, robust_shape);
//    /// Undo similarity transform
//    for (TrainingSample &sample : samples)
//    {
//      applyRigidTransformToShape(rigids[sample.sample_idx], center, sample.current_shape);
//      applyRigidTransformToShape(rigids[sample.sample_idx], center, sample.target_shape);
//    }
    /// Compute training samples error
    computeError(anns, utform, scales, measure, samples, train_means[i], train_variances[i]);
    UPM_PRINT("[" << i << "/" << NUM_LEVELS << "] Training: " << train_means[i] << " (±" << train_variances[i] << ")");
  }

  UPM_PRINT("Training complete");
  return EnsembleTrees(SHAPE_SIZE, robust_shape, robust_label, forests);
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
void
EnsembleTrainer::computeError
  (
  const std::vector<FaceAnnotation> &anns,
  const std::vector<cv::Mat> &tform,
  const std::vector<float> &scales,
  const ErrorMeasure &measure,
  std::vector<TrainingSample> &samples,
  float &train_mean,
  float &train_variance
  ) const
{
  /// Compute samples error
  train_mean = 0.0f, train_variance = 0.0f;
  for (TrainingSample &sample : samples)
  {
    FaceAnnotation current;
    shapeToFaceParts(sample.current_shape, sample.current_label, tform[sample.image_idx], scales[sample.image_idx], current.parts);
    std::vector<unsigned int> indices;
    std::vector<float> errors;
    getNormalizedErrors(current, anns[sample.image_idx], measure, indices, errors);
    sample.error = static_cast<float>(std::accumulate(errors.begin(),errors.end(),0.0)) / static_cast<float>(errors.size());
    train_mean += sample.error;
    train_variance += (sample.error * sample.error);
  }
  train_mean /= static_cast<float>(samples.size());
  train_variance = (train_variance/samples.size()) - (train_mean*train_mean);
};

} // namespace upm
