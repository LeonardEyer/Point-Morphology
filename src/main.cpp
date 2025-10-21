#include <polyscope/types.h>
#include <polyscope/utilities.h>
#include <polyscope/point_cloud.h>
#include <polyscope/polyscope.h>

int main() {
  
  // Options
  polyscope::options::autocenterStructures = true;
  polyscope::view::windowWidth = 1024;
  polyscope::view::windowHeight = 1024;

  // Initialize polyscope
  polyscope::init();
  
  std::vector<glm::vec3> points;

  for (auto i = 0; i < 3000; i++) {
    points.push_back(glm::vec3{polyscope::randomUnit() - .5,
                               polyscope::randomUnit() - .5,
                               polyscope::randomUnit() - .5});
  }

  auto *psCloud = polyscope::registerPointCloud("some points", points);
  psCloud->setPointRadius(0.004);
  psCloud->setPointRenderMode(polyscope::PointRenderMode::Sphere);

  polyscope::show();
}
