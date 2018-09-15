#include <gtest/gtest.h>
#include <contrast.hpp>

TEST(ContrastTests, RealImage)
{
  const std::string testImagePath("../tests/testimage.jpg");
  cv::Mat colorTestImage = cv::imread(testImagePath);
  ASSERT_TRUE(colorTestImage.data);
  cv::Mat testImage;
  cv::cvtColor(colorTestImage, testImage, cv::COLOR_RGB2GRAY);

  double metricValue = Contrast::calculate(testImage);
  EXPECT_TRUE(metricValue != 0.0); // Really weak test TODO
}
