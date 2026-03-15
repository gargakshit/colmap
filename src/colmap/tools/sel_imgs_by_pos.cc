#include "colmap/controllers/option_manager.h"
#include "colmap/scene/reconstruction.h"
#include <filesystem>

using namespace colmap;

int main(int argc, char** argv) {
  OptionManager options;
  std::string input_path;
  std::string output_path;
  int center_view_id = static_cast<int>(kInvalidImageId);
  double radius;
  std::string pose_prior_path = "";
  options.AddImageOptions();
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddRequiredOption("center_view_id", &center_view_id);
  options.AddRequiredOption(
      "radius", &radius, "Radius to select images around the center view, unit meter");
  options.AddDefaultOption("pose_prior_path",
                           &pose_prior_path,
                           "Whether to copy pose prior files together with image files");
  options.Parse(argc, argv);

  if (!std::filesystem::is_directory(input_path)) {
    LOG(ERROR) << "`input_path` is not a directory";
    return EXIT_FAILURE;
  }

  if (!std::filesystem::is_directory(output_path)) {
    LOG(ERROR) << "`output_path` is not a directory";
    return EXIT_FAILURE;
  }
  CHECK_GT(radius, 0.0) << "Radius must be greater than 0";
  CHECK_NE(static_cast<image_t>(center_view_id), kInvalidImageId);

  const bool copy_pose_priors = !pose_prior_path.empty();

  std::filesystem::create_directories(
      std::filesystem::path(output_path) / "images");
  if (copy_pose_priors) {
    std::filesystem::create_directories(
        std::filesystem::path(output_path) / "pose_priors");
  }

  auto reconstruction = std::make_shared<Reconstruction>();
  reconstruction->Read(input_path);

  const std::vector<image_t>& reg_image_ids = reconstruction->RegImageIds();
  const Eigen::Vector3d center_view_position =
      reconstruction->Image(static_cast<image_t>(center_view_id)).ProjectionCenter();

  for (const image_t image_id : reg_image_ids) {
    const Image& image = reconstruction->Image(image_id);
    const double distance = (center_view_position - image.ProjectionCenter()).norm();
    if (distance > radius) {
      continue;
    }

    LOG(INFO) << "Copying " << image_id << " , distance " << distance;
    const std::string& image_name = image.Name();
    const std::filesystem::path src_path =
        *options.image_path.get() / image_name;
    const std::filesystem::path dst_path =
        std::filesystem::path(output_path) / "images" / image_name;

    std::filesystem::create_directories(dst_path.parent_path());
    std::filesystem::copy_file(
        src_path, dst_path, std::filesystem::copy_options::overwrite_existing);
    if (copy_pose_priors) {
      const std::string csv_file =
          std::filesystem::path(image_name).stem().string() + ".csv";
      const std::filesystem::path prior_src =
          std::filesystem::path(pose_prior_path) / csv_file;
      const std::filesystem::path prior_dst =
          std::filesystem::path(output_path) / "pose_priors" / csv_file;
      if (std::filesystem::exists(prior_src)) {
        std::filesystem::create_directories(prior_dst.parent_path());
        std::filesystem::copy_file(
            prior_src,
            prior_dst,
            std::filesystem::copy_options::overwrite_existing);
      } else {
        LOG(WARNING) << "csv file " << csv_file << " doesn't exist";
      }
    }
  }

  return EXIT_SUCCESS;
}
