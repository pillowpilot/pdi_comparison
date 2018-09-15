#include <gtest/gtest.h>
#include <psnr.hpp>
#include <opencv2/opencv.hpp>
#include <cmath>

TEST(PSNRTest, SameImage)
{
  const std::string testImagePath("../tests/testimage.jpg");
  cv::Mat colorTestImage = cv::imread(testImagePath);
  ASSERT_TRUE(colorTestImage.data);
  cv::Mat testImage;
  cv::cvtColor(colorTestImage, testImage, cv::COLOR_RGB2GRAY);

  double metricValue = PSNR::calculate(testImage, testImage);
  EXPECT_TRUE(std::isinf(metricValue));
}

TEST(PSNRTest, ImageVsPlusOne)
{
  const std::string testImagePath("../tests/testimage.jpg");
  cv::Mat colorTestImage = cv::imread(testImagePath);
  ASSERT_TRUE(colorTestImage.data);
  cv::Mat testImageA;
  cv::cvtColor(colorTestImage, testImageA, cv::COLOR_RGB2GRAY);

  cv::Mat testImageB = testImageA.clone();
  cv::MatIterator_<uchar> iter = testImageB.begin<uchar>();
  while(iter!=testImageB.end<uchar>())
    {
      *iter += 1;
      ++iter;
    }

  double metricValue = PSNR::calculate(testImageA, testImageB);
  EXPECT_TRUE(std::isfinite(metricValue));
}
