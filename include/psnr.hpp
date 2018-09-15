#ifndef PSNR_H
#define PSNR_H

#include <opencv2/opencv.hpp>
#include <limits>
#include <cassert>
#include <iostream>

class PSNR
{
private:
  static double calculateMSE(const cv::Mat& imageA, const cv::Mat& imageB);
public:
  static double calculate(const cv::Mat& imageA, const cv::Mat& imageB); 
};

double PSNR::calculateMSE(const cv::Mat& imageA, const cv::Mat& imageB)
{
  double mse = 0;
  for(int i = 0; i < imageA.rows; i++)
    for(int j = 0; j < imageA.cols; j++)
      {
	const auto valueA = imageA.at<uchar>(i, j);
	const auto valueB = imageB.at<uchar>(i, j);
	mse += std::pow(valueA - valueB, 2);
      }
  mse /= imageA.rows * imageA.cols;
  return mse;
}

double PSNR::calculate(const cv::Mat& imageA, const cv::Mat& imageB)
{
  assert(imageA.rows == imageB.rows);
  assert(imageA.cols == imageB.cols);

  const auto mse = calculateMSE(imageA, imageB);
  if( mse < 1e-9 )
    return std::numeric_limits<double>::infinity();
      
  const double psnr = 10*std::log10(std::pow(255, 2) / mse);
  return psnr;
}

#endif /* PSNR_H */
