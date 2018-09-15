#ifndef ENTROPY_H
#define ENTROPY_H

#include <opencv2/opencv.hpp>
#include <cmath>
#include <Utils.hpp>

class Entropy
{
private:
  static cv::Mat calculateNormalizedHistogram(const cv::Mat& image);
public:
  static double calculate(const cv::Mat& image);
};

cv::Mat Entropy::calculateNormalizedHistogram(const cv::Mat& image)
{
  cv::Mat histogram(1, 256, CV_64F, cv::Scalar(0));

  {
    cv::MatConstIterator_<uchar> iter = image.begin<uchar>();
    while(iter != image.end<uchar>())
      {
	histogram.at<double>(*iter)++;
	++iter;
      }
  }

  const double area = image.rows * image.cols;

  {
    cv::MatIterator_<double> iter = histogram.begin<double>();
    while(iter != histogram.end<double>())
      {
	*iter = *iter / area;
	++iter;
      }
  }
  return histogram;
}


double Entropy::calculate(const cv::Mat& image)
{
  const auto normalizedHistogram = calculateNormalizedHistogram(image);

  double entropy = 0;
  cv::MatConstIterator_<double> iter = normalizedHistogram.begin<double>();
  while(iter != normalizedHistogram.end<double>())
    {
      if( *iter != 0.0 )
	entropy += *iter * std::log2(*iter);      
      ++iter;
    }

  return -entropy;
}


#endif /* ENTROPY_H */
