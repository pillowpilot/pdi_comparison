#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <entropy.hpp>
#include <iostream>

TEST(EntropyTests, OneIntensityImage)
{
  const int intensity = 10;
  cv::Mat testImage(1000, 1000, CV_8UC1, cv::Scalar(intensity));

  double metricValue = Entropy::calculate(testImage);
  EXPECT_DOUBLE_EQ(metricValue, 0.0);
}

TEST(EntropyTest, RealImage)
{
  const std::string testImagePath("../tests/testimage.jpg");
  cv::Mat colorTestImage = cv::imread(testImagePath);
  ASSERT_TRUE(colorTestImage.data);
  cv::Mat testImage;
  cv::cvtColor(colorTestImage, testImage, cv::COLOR_RGB2GRAY);

  double metricValue = Entropy::calculate(testImage);
  //std::cout << metricValue << "\n";
  EXPECT_TRUE(metricValue != 0.0);
}
