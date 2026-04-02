#include "PointStructuringElement.hpp"
#include "glm/ext/scalar_constants.hpp"
#include "glm/ext/vector_float3.hpp"

#include <gtest/gtest.h>
#include <limits>

using namespace structuring_elements;

static constexpr auto eps = std::numeric_limits<float>::epsilon() * 10;

TEST(PointStructuringElementTests, TestProjection) {

  for (auto s = 1; s < 9; s++) {
    const auto c = glm::vec3(0, 0, 0);
    const auto B = sdf::sphere;

    const auto pse = PointStructuringElement{static_cast<float>(s), c, B};

    const auto p = glm::vec3(10, 0, 0);

    EXPECT_NEAR(pse.distance(p), 10 - s, eps);
    EXPECT_EQ(pse.gradient(p), glm::vec3(1, 0, 0));
    EXPECT_EQ(project(pse, p), glm::vec3(s, 0, 0));
  }

  static const auto edge_length = 1.f / std::sqrt(3);
  for (auto s = 1; s < 9; s++) {
    const auto c = glm::vec3(0, 0, 0);
    const auto B = sdf::cube;

    const auto pse = PointStructuringElement{static_cast<float>(s), c, B};

    const auto p = glm::vec3(10, 0, 0);

    EXPECT_NEAR(pse.distance(p), 10 - (s * edge_length), eps);
    EXPECT_EQ(pse.gradient(p), glm::vec3(1, 0, 0));

    const auto projected = project(pse, p);

    EXPECT_NEAR(projected.x, s * edge_length, eps);
    EXPECT_NEAR(projected.y, 0, eps);
    EXPECT_NEAR(projected.z, 0, eps);
  }
}
