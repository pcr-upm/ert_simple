/** ****************************************************************************
 *  @file    LearningAlgorithmSimple.cpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2017/06
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ----------------------- INCLUDES --------------------------------------------
#include <LearningAlgorithmSimple.hpp>
#include <utils.hpp>
#include <transformation.hpp>
#include <omp.h>

namespace upm {

const unsigned int NUM_SUBLEVELS = 500; // K
const unsigned int NUM_TEST_SPLITS = 20; // S
const unsigned int TREE_DEPTH = 5;
const float SPLIT_THRESHOLD = 0.3;
const float NU = 0.1; // ν

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
LearningAlgorithmSimple::robustEstimation
  (
  const std::vector<cv::Mat> &shapes,
  const std::vector<cv::Mat> &labels,
  cv::Mat &robust_shape,
  cv::Mat &robust_label
  )
{
  const unsigned int num_shapes = static_cast<int>(shapes.size());
  if (num_shapes == 0) return;

  /// Mean shape using squared error loss
  for (unsigned int i=0; i < num_shapes; i++)
  {
    robust_shape += shapes[i];
    robust_label += labels[i];
  }
  robust_shape.col(0) /= robust_label;
  robust_shape.col(1) /= robust_label;
  robust_shape.col(2) /= cv::Mat(robust_shape.rows,1,robust_shape.type(),cv::Scalar(num_shapes));
  robust_label.col(0) /= robust_label;
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
LearningAlgorithmSimple::learnRegressor
  (
  std::vector<TrainingSample> &samples,
  EnsembleTrees &forest,
  std::vector<std::vector<int>> &landmark_per_tree,
  const bool coarse_to_fine,
  const cv::Mat &center,
  const cv::Mat &robust_shape
  )
{
  /// One part with all the facial landmarks
  const unsigned int num_parts = 1;
  std::vector< std::vector<int> > parts(num_parts), parts_idx(num_parts);
  for (const std::pair<FacePartLabel,std::vector<int>> &part : DB_PARTS)
    parts[0].insert(parts[0].end(), part.second.begin(), part.second.end());
  /// Keep extracted features landmarks order
  for (const std::vector<int> &part : parts)
    for (const int &idx : part)
      parts_idx[0].push_back(static_cast<int>(std::distance(parts[0].begin(), std::find(parts[0].begin(),parts[0].end(),idx))));

  _num_leaf_nodes = static_cast<unsigned int>(std::pow(2.0f,static_cast<float>(TREE_DEPTH)));
  _num_split_nodes = _num_leaf_nodes - 1;
  _num_nodes = _num_leaf_nodes + _num_split_nodes;

  /// Ensemble of weak regression trees trained using gradient boosting
  forest.resize(NUM_SUBLEVELS, std::vector<impl::RegressionTree>(num_parts));
  for (unsigned int i=0; i < NUM_SUBLEVELS; i++)
    for (unsigned int j=0; j < num_parts; j++)
      forest[i][j] = makeRegressionTree(samples, parts_idx[j]);
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
impl::RegressionTree
LearningAlgorithmSimple::makeRegressionTree
  (
  std::vector<TrainingSample> &samples,
  const std::vector<int> &part_idx
  )
{
  const unsigned int num_samples = static_cast<int>(samples.size());
  const unsigned int num_part_landmarks = static_cast<int>(part_idx.size());
  std::vector<cv::Mat> residuals_part(num_samples), Ws_part(num_samples), gradients_part(num_samples);
  #pragma omp parallel for
  for (unsigned int i=0; i < num_samples; i++)
  {
    /// Compute residual r for each sample to minimize loss function
    cv::Mat current = samples[i].current_shape.clone();
    cv::Mat target  = samples[i].target_shape.clone();
    cv::Mat W = samples[i].target_label.mul(samples[i].current_label);
    cv::Mat residual = cv::Mat::diag(W) * cv::Mat(target-current);
    residuals_part[i] = cv::Mat(num_part_landmarks,3,CV_32FC1);
    Ws_part[i] = cv::Mat(num_part_landmarks,1,CV_32FC1);
    for (unsigned int j=0; j < num_part_landmarks; j++)
    {
      residual.row(part_idx[j]).copyTo(residuals_part[i].row(j));
      W.row(part_idx[j]).copyTo(Ws_part[i].row(j));
    }
    /// Compute the negative gradient related for each residual
    gradients_part[i] = residuals_part[i].clone();
  }

  /// Compute the sum gradients for all the samples
  std::vector<cv::Mat> sum_Ws(_num_nodes), sum_gradients(_num_nodes); // split_nodes + leaf_nodes
  std::vector< std::vector<int> > node_indices(_num_nodes);
  sum_Ws[0] = cv::Mat::zeros(num_part_landmarks,1,CV_32FC1);
  sum_gradients[0] = cv::Mat::zeros(num_part_landmarks,3,CV_32FC1);
  for (int i=0; i < num_samples; i++)
  {
    sum_Ws[0] += Ws_part[i].clone();
    sum_gradients[0] += gradients_part[i].clone();
    node_indices[0].push_back(i);
  }

  impl::RegressionTree tree;
  tree.splits.clear();
  tree.splits.resize(_num_split_nodes); // split_nodes
  for (unsigned int i=0; i < _num_split_nodes; i++)
    tree.splits[i] = generateBestSplit(samples, gradients_part, Ws_part, part_idx, i, node_indices, sum_gradients, sum_Ws);

  /// Compute each leaf node robust residual and apply shrinkage ν to avoid over-fitting
  std::vector< std::vector<cv::Mat> > residuals_leaf(_num_leaf_nodes), Ws_leaf(_num_leaf_nodes);
  for (unsigned int i=0; i < num_samples; i++)
  {
    unsigned int leaf_idx = tree.predict(samples[i].features);
    residuals_leaf[leaf_idx].push_back(residuals_part[i]);
    Ws_leaf[leaf_idx].push_back(Ws_part[i]);
  }
  std::vector<cv::Mat> robust_residual(_num_leaf_nodes), robust_W(_num_leaf_nodes);
  for (unsigned int i=0; i < _num_leaf_nodes; i++)
  {
    robust_residual[i] = cv::Mat::zeros(num_part_landmarks,3,CV_32FC1);
    robust_W[i] = cv::Mat::zeros(num_part_landmarks,1,CV_32FC1);
    robustEstimation(residuals_leaf[i], Ws_leaf[i], robust_residual[i], robust_W[i]);
    robust_residual[i] *= NU;
  }

  tree.leafs.clear();
  tree.leafs.resize(_num_leaf_nodes); // leaf_nodes
  const unsigned int num_landmarks = static_cast<unsigned int>(samples[0].current_shape.rows);
  for (unsigned int i=0; i < _num_leaf_nodes; i++)
  {
    tree.leafs[i].residual = cv::Mat::zeros(num_landmarks,3,CV_32FC1);
    for (unsigned int j=0; j < num_part_landmarks; j++)
      robust_residual[i].row(j).copyTo(tree.leafs[i].residual.row(part_idx[j]));
    /// Update each sample using their residual predicted
    for (int j : node_indices[_num_split_nodes+i])
      addResidualToShape(tree.leafs[i].residual, samples[j].current_shape);
  }
  return tree;
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
impl::SplitFeature
LearningAlgorithmSimple::generateBestSplit
  (
  const std::vector<TrainingSample> &samples,
  const std::vector<cv::Mat> &gradients,
  const std::vector<cv::Mat> &Ws,
  const std::vector<int> &part_idx,
  unsigned int split_idx,
  std::vector< std::vector<int> > &node_indices,
  std::vector<cv::Mat> &sum_gradients,
  std::vector<cv::Mat> &sum_Ws
  )
{
  /// Check number of samples is not zero
  if (node_indices[split_idx].empty()) return impl::SplitFeature();

  /// Generate a bunch of random splits θ and return the best one
  std::vector<impl::SplitFeature> feats(NUM_TEST_SPLITS);
  const int num_sampling = samples[node_indices[split_idx][0]].features.cols;
  std::vector<float> rnd_threshold(NUM_TEST_SPLITS);
  for (unsigned int i=0; i < NUM_TEST_SPLITS; i++)
  {
    feats[i].type = _split_type;
    feats[i].lnd = static_cast<unsigned int>(part_idx[_rnd.uniform(0,static_cast<int>(part_idx.size()))]);
    feats[i].desc1 = static_cast<unsigned int>(_rnd.uniform(0,num_sampling));
    feats[i].desc2 = static_cast<unsigned int>(_rnd.uniform(0,num_sampling-1));
    if (feats[i].desc2 >= feats[i].desc1)
      feats[i].desc2++;
    rnd_threshold[i] = _rnd.uniform(0.0f,1.0f);
  }
  #pragma omp parallel for
  for (unsigned int i=0; i < NUM_TEST_SPLITS; i++)
  {
    float desc, min_desc = std::numeric_limits<float>::max(), max_desc = std::numeric_limits<float>::min();
    for (int j : node_indices[split_idx])
    {
      desc = feats[i].getDescriptor(samples[j].features);
      if (desc < min_desc)
        min_desc = desc;
      if (desc > max_desc)
        max_desc = desc;
    }
    feats[i].thresh = (rnd_threshold[i] * (max_desc-min_desc) + min_desc) * SPLIT_THRESHOLD;
  }

  std::vector< std::vector<int> > left_indices(NUM_TEST_SPLITS), right_indices(NUM_TEST_SPLITS);
  /// Compute the sum residual ∑r for samples that go to left and right
  std::vector<cv::Mat> left_gradients(NUM_TEST_SPLITS), left_Ws(NUM_TEST_SPLITS);
  #pragma omp parallel for
  for (unsigned int i=0; i < NUM_TEST_SPLITS; i++)
  {
    left_gradients[i] = cv::Mat::zeros(sum_gradients[0].size(),sum_gradients[0].type());
    left_Ws[i] = cv::Mat::zeros(sum_Ws[0].size(),sum_Ws[0].type());
    for (int j : node_indices[split_idx])
    {
      if (feats[i].getDescriptor(samples[j].features) > feats[i].thresh)
      {
        left_indices[i].push_back(j);
        left_gradients[i] += gradients[j].clone();
        left_Ws[i] += Ws[j].clone();
      }
      else
        right_indices[i].push_back(j);
    }
  }

  /// Choose split which minimize entropy E(Q,θ) = max((∑r*∑r)/|Q|)
  std::vector<float> scores(NUM_TEST_SPLITS);
  #pragma omp parallel for
  for (unsigned int i=0; i < NUM_TEST_SPLITS; i++)
  {
    // Use 'x' and 'y' coordinates to compute the score
    cv::Mat gradients_i, left_gradients_i, Ws_i, left_Ws_i;
    gradients_i.push_back(sum_gradients[split_idx].col(0));
    gradients_i.push_back(sum_gradients[split_idx].col(1));
    left_gradients_i.push_back(left_gradients[i].col(0));
    left_gradients_i.push_back(left_gradients[i].col(1));
    cv::Mat right_gradients_i = gradients_i - left_gradients_i;
    Ws_i.push_back(sum_Ws[split_idx]);
    Ws_i.push_back(sum_Ws[split_idx]);
    left_Ws_i.push_back(left_Ws[i]);
    left_Ws_i.push_back(left_Ws[i]);
    cv::Mat right_Ws_i = Ws_i - left_Ws_i;
    cv::Mat left_entropy  = left_gradients_i.t() * cv::Mat::diag(1.0f/left_Ws_i) * left_gradients_i;
    cv::Mat right_entropy = right_gradients_i.t() * cv::Mat::diag(1.0f/right_Ws_i) * right_gradients_i;
    scores[i] = left_entropy.at<float>(0,0) + right_entropy.at<float>(0,0);
  }
  unsigned int best_feat = static_cast<unsigned int>(std::distance(scores.begin(), std::max_element(scores.begin(),scores.end())));
//  float best_score = *std::max_element(scores.begin(), scores.end());

  /// Return optimal left and right children
  sum_gradients[(2*split_idx)+1] = left_gradients[best_feat];
  sum_gradients[(2*split_idx)+2] = sum_gradients[split_idx] - left_gradients[best_feat];
  sum_Ws[(2*split_idx)+1] = left_Ws[best_feat];
  sum_Ws[(2*split_idx)+2] = sum_Ws[split_idx] - left_Ws[best_feat];

  node_indices[(2*split_idx)+1] = left_indices[best_feat];
  node_indices[(2*split_idx)+2] = right_indices[best_feat];
//  std::cout << feats[best_feat].desc1 << " " << feats[best_feat].desc2 << " " << feats[best_feat].thresh << std::endl;
  return feats[best_feat];
};

} // namespace upm
