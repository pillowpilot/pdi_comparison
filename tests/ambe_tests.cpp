#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <ambe.hpp>

TEST(AmbeTests, SameImage)
{
  const std::string testImagePath("../tests/testimage.jpg");
  cv::Mat colorTestImage = cv::imread(testImagePath);
  ASSERT_TRUE(colorTestImage.data);
  cv::Mat testImage;
  cv::cvtColor(colorTestImage, testImage, cv::COLOR_RGB2GRAY);

  double metricValue = AMBE::calculate(testImage, testImage);
  EXPECT_DOUBLE_EQ(metricValue, 0.0);
}

TEST(AmbeTests, ImageVsPlusOne)
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

  double metricValue = AMBE::calculate(testImageA, testImageB);
  EXPECT_TRUE(metricValue != 0.0);
}
