#include "VoxelGrid.hpp"

#include <gtest/gtest.h>

TEST(VoxelGridTests, ScalarVoxelGrid) {
  auto grid = VoxelGrid<float>({5, 5, 5});

  grid.insert({0, 0, 0}, 0.2f);
  EXPECT_EQ(grid.get({0, 0, 0}), 0.2f);
  EXPECT_EQ(grid.get({0.1, 0.5, 2.3}), 0.2f);

  // outside
  EXPECT_EQ(grid.get({5.1, 0.5, 2.3}), std::nullopt);

  grid.insert({5, 0, 2}, 1.2f);

  EXPECT_EQ(grid.get({5.1, 0.5, 2.3}), 1.2f);
}
