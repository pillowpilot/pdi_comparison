#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <bhe3pl.hpp>

TEST(BHE3PLTests, HistogramSegmentation)
{
  const std::string testImagePath("../tests/testimage.jpg");
  cv::Mat colorTestImage = cv::imread(testImagePath);
  ASSERT_TRUE(colorTestImage.data);
  cv::Mat testImage;
  cv::cvtColor(colorTestImage, testImage, cv::COLOR_RGB2GRAY);

  cv::Mat histogram(1, 256, CV_8UC1, cv::Scalar(0));
  cv::MatIterator_<uchar> iter = testImage.begin<uchar>();
  while( iter != testImage.end<uchar>() )
    {
      histogram.at<uchar>(*iter)++;
      ++iter;
    }

  std::cout << "Histogram: " << histogram << "\n";

  BHE3PL algorithm(histogram);
  std::cout << "G SP: " << algorithm.getGlobalSeparatingPoint() << "\n";
  std::cout << "SubH Low: " << algorithm.getSubHistogramLow() << "\n";
  std::cout << "SubH High: " << algorithm.getSubHistogramHigh() << "\n";
}

TEST(BHE3PL, HistogramModification)
{
  const std::string testImagePath("../tests/testimage.jpg");
  cv::Mat colorTestImage = cv::imread(testImagePath);
  ASSERT_TRUE(colorTestImage.data);
  cv::Mat testImage;
  cv::cvtColor(colorTestImage, testImage, cv::COLOR_RGB2GRAY);

  cv::Mat histogram(1, 256, CV_8UC1, cv::Scalar(0));
  cv::MatIterator_<uchar> iter = testImage.begin<uchar>();
  while( iter != testImage.end<uchar>() )
    {
      histogram.at<uchar>(*iter)++;
      ++iter;
    }

  std::cout << "Histogram: " << histogram << "\n";

  BHE3PL algorithm(histogram);
  std::cout << "Low SP: " << algorithm.getLowSeparatingPoint() << "\n";
  std::cout << "High SP: " << algorithm.getHighSeparatingPoint() << "\n";
  std::cout << "Low PL: " << algorithm.getLowPlateauLimits() << "\n";
  std::cout << "High PL: " << algorithm.getHighPlateauLimits() << "\n";
  std::cout << "Modified: " << algorithm.getModifiedHistogram() << "\n";
}
