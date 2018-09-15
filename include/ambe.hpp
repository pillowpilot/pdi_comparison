#ifndef AMBE_H
#define AMBE_H

#include <opencv2/opencv.hpp>
#include <cmath>

class AMBE
{
public:
  static cv::Mat calculateHistogram(const cv::Mat& image);
  static double calculateMean(const cv::Mat& histogram, const int imageArea);
  static double calculate(const cv::Mat& imageA, const cv::Mat& imageB);
};

cv::Mat AMBE::calculateHistogram(const cv::Mat& image)
{
  cv::Mat histogram(1, 256, CV_8UC1, cv::Scalar(0));

  cv::MatConstIterator_<uchar> iter = image.begin<uchar>();
  while(iter != image.end<uchar>())
    {
      histogram.at<uchar>(*iter)++;
      ++iter;
    }
  
  return histogram;
}

double AMBE::calculateMean(const cv::Mat& histogram, const int imageArea)
{
  double mean = 0;
  for(int intensity = 0; intensity < histogram.cols; intensity++)
    mean += (double)intensity * histogram.at<uchar>(intensity) / imageArea;

  return mean;
}

double AMBE::calculate(const cv::Mat& imageA, const cv::Mat& imageB)
{
  const auto histogramA = calculateHistogram(imageA);
  const auto imageAreaA = imageA.rows * imageA.cols;
  
  const auto histogramB = calculateHistogram(imageB);
  const auto imageAreaB = imageB.rows * imageB.cols;

  const auto meanBrighnessA = calculateMean(histogramA, imageAreaA);
  const auto meanBrighnessB = calculateMean(histogramB, imageAreaB);

  return std::abs(meanBrighnessA - meanBrighnessB);
}

#endif /* AMBE_H */
