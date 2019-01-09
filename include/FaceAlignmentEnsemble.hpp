/** ****************************************************************************
 *  @file    FaceAlignmentEnsemble.hpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2017/06
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef FACE_ALIGNMENT_ENSEMBLE_HPP
#define FACE_ALIGNMENT_ENSEMBLE_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <FaceAlignment.hpp>
#include <opencv2/opencv.hpp>
#include <EnsembleTrees.hpp>

namespace upm {

/** ****************************************************************************
 * @class FaceAlignmentEnsemble
 * @brief Class used for facial feature point detection.
 ******************************************************************************/
class FaceAlignmentEnsemble: public FaceAlignment
{
public:
  FaceAlignmentEnsemble(std::string path) : _path(path) {};

  ~FaceAlignmentEnsemble() {};

  void
  parseOptions
    (
    int argc,
    char **argv
    );

  void
  train
    (
    const std::vector<upm::FaceAnnotation> &anns_train,
    const std::vector<upm::FaceAnnotation> &anns_valid
    );

  void
  load();

  void
  process
    (
    cv::Mat frame,
    std::vector<FaceAnnotation> &faces,
    const FaceAnnotation &ann
    );

private:
  std::string _path;
  EnsembleTrees _model;
};

} // namespace upm

#endif /* FACE_ALIGNMENT_ENSEMBLE_HPP */
