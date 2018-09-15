#ifndef CONTRAST_H
#define CONTRAST_H

#include <opencv2/opencv.hpp>
#include <ambe.hpp>
#include <Utils.hpp>

class Contrast
{
public:
  static double calculate(const cv::Mat& image);
};

double Contrast::calculate(const cv::Mat& image)
{
  const auto histogram = AMBE::calculateHistogram(image);
  const double imageArea = image.rows * image.cols;
  const auto meanBrighness = AMBE::calculateMean(histogram, imageArea);

  cv::Mat normalizedHistogram(1, histogram.cols, CV_64F, cv::Scalar(0));

  for(int intensity; intensity < normalizedHistogram.cols; intensity++)
    {
      normalizedHistogram.at<double>(0, intensity) =
	histogram.at<uchar>(0, intensity) / imageArea;
    }

  double contrast = 0;
  for(int intensity = 0; intensity < histogram.cols; intensity++)
    {
      contrast += std::pow(intensity - meanBrighness, 2) *
	normalizedHistogram.at<double>(intensity);
    }

  contrast = std::sqrt(contrast);

  return contrast;
}


#endif /* CONTRAST_H */
