/** ****************************************************************************
 *  @file    FaceAlignmentEnsemble.cpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2017/06
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ----------------------- INCLUDES --------------------------------------------
#include <FaceAlignmentEnsemble.hpp>
#include <trace.hpp>
#include <utils.hpp>
#include <EnsembleTrainer.hpp>
#include <cereal/cereal.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/vector.hpp>
#include <boost/progress.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <fstream>

namespace upm {

const unsigned int MAX_IMAGES = 4000;
const float FACE_HEIGHT = 160.0f;

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
FaceAlignmentEnsemble::parseOptions
  (
  int argc,
  char **argv
  )
{
  /// Declare the supported program options
  FaceAlignment::parseOptions(argc, argv);
  namespace po = boost::program_options;
  po::options_description desc("FaceAlignmentEnsemble options");
  UPM_PRINT(desc);
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
FaceAlignmentEnsemble::train
  (
  const std::vector<upm::FaceAnnotation> &anns_train,
  const std::vector<upm::FaceAnnotation> &anns_valid
  )
{
  /// Use all annotations for train
  std::vector<upm::FaceAnnotation> anns;
  anns.insert(anns.end(), anns_train.begin(), anns_train.end());
  anns.insert(anns.end(), anns_valid.begin(), anns_valid.end());
  EnsembleTrainer trainer;

  /// Select annotations
  const unsigned int num_data = static_cast<int>(MIN(MAX_IMAGES, anns.size()));
  std::vector<FaceAnnotation> data;
  data.insert(data.end(), anns.begin(), anns.begin()+num_data);
  UPM_PRINT("Total number of images: " << num_data);

  /// Load images and resize images
  std::vector<cv::Mat> imgs(num_data);
  std::vector<float> scales(num_data);
  boost::progress_display show_progress(num_data);
  for (int i=0; i < num_data; i++, ++show_progress)
  {
    imgs[i] = cv::imread(data[i].filename, cv::IMREAD_COLOR);
    scales[i] = FACE_HEIGHT/data[i].bbox.pos.height;
    cv::resize(imgs[i], imgs[i], cv::Size(), scales[i], scales[i]);
  }

  /// Now finally generate the shape model and save the model to disk
  _model = trainer.train(data, imgs, scales, _measure);
  try
  {
    std::string filename = "model.bin";
    std::ofstream ofs(_path + _database + "/" + filename);
    cereal::BinaryOutputArchive oa(ofs);
    oa << _model << DB_PARTS << DB_LANDMARKS;
    ofs.flush();
    ofs.close();
    UPM_PRINT("Complete predictor saved: " << _path + _database + "/" + filename);
  }
  catch (cereal::Exception &ex)
  {
    UPM_ERROR("Exception during predictor serialization: " << ex.what());
  }
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
FaceAlignmentEnsemble::load()
{
  /// Loading shape predictors
  UPM_PRINT("Loading facial-feature predictors");
  std::vector<std::string> paths;
  boost::filesystem::path dir_path(_path + _database + "/");
  boost::filesystem::directory_iterator end_it;
  for (boost::filesystem::directory_iterator it(dir_path); it != end_it; ++it)
    paths.push_back(it->path().string());
  sort(paths.begin(), paths.end());
  UPM_PRINT("> Number of predictors found: " << paths.size());

  for (const std::string &path : paths)
  {
    try
    {
      std::ifstream ifs(path);
      cereal::BinaryInputArchive ia(ifs);
      ia >> _model >> DB_PARTS >> DB_LANDMARKS;
      ifs.close();
    }
    catch (cereal::Exception &ex)
    {
      UPM_ERROR("Exception during predictor deserialization: " << ex.what());
    }
  }
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
FaceAlignmentEnsemble::process
  (
  cv::Mat frame,
  std::vector<FaceAnnotation> &faces,
  const FaceAnnotation &ann
  )
{
  /// Analyze each detected face to tell us the face part locations
  for (FaceAnnotation &face : faces)
  {
    float scale = FACE_HEIGHT/face.bbox.pos.height;
    cv::Mat img;
    cv::resize(frame, img, cv::Size(), scale, scale);
    _model.process(img, scale, face, ann);
  }
};

} // namespace upm
