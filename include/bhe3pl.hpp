#ifndef BHE3PL_H
#define BHE3PL_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <Utils.hpp>

class BHE3PL
{
private:
  using MatPair = std::pair<cv::Mat, cv::Mat>;
  using VectorDouble = std::vector<double>;

  cv::Mat _globalHistogram, _normalizedGlobalHistogram, _subHistogramLow, _subHistogramHigh;
  cv::Mat _modifiedGlobalHistogram, _modifiedLowSubHistogram, _modifiedHighSubHistogram;
  cv::Mat _finalTransformation;
  
  double _globalSeparatingPoint, _lowSeparatingPoint, _highSeparatingPoint;
  VectorDouble _lowGrayLevelRatios, _highGrayLevelRatios;
  VectorDouble _lowPlateauLimits, _highPlateauLimits;
  int _numberOfBins;

  // Commons
  cv::Mat normalizeHistogram(const cv::Mat& histogram);
  cv::Mat calculateCummulative(const cv::Mat& histogram);

  // First stage: Histogram Segmentation
  double calculateSeparatingPoint(const cv::Mat& normalizedHistogram);
  void buildSubHistograms();
  void applyHistogramSegmentation();

  // Second stage: Histogram Modification
  void calculateSubHistogramSeparatingPoints();
  void calculateGrayLevels1And3(const double grayLevel2, double* grayLevel1, double* grayLevel3);
  void calculateGrayLevelRatios();
  void calculatePlateauLimits();  
  void calculateModifiedHistogram(const cv::Mat& subHistogram, const VectorDouble& plateauLimits, cv::Mat& modifiedHistogram);
  void applyHistogramModification();

  // Third stage: Histogram Transformation
  void applyHistogramTransformation();
    
public:  
  BHE3PL(const cv::Mat& histogram)
    :_globalHistogram(histogram), _numberOfBins(histogram.cols)
  {
    applyHistogramSegmentation();
    applyHistogramModification();
    applyHistogramTransformation();
  }
  double getGlobalSeparatingPoint() const { return _globalSeparatingPoint; }
  double getLowSeparatingPoint() const { return _lowSeparatingPoint; }
  double getHighSeparatingPoint() const { return _highSeparatingPoint; }
  VectorDouble getLowPlateauLimits() const { return _lowPlateauLimits; }
  VectorDouble getHighPlateauLimits() const { return _highPlateauLimits; }
  cv::Mat getSubHistogramLow() const { return _subHistogramLow; }
  cv::Mat getSubHistogramHigh() const { return _subHistogramHigh; }
  cv::Mat getModifiedHistogram() const { return _modifiedGlobalHistogram; }
};

cv::Mat BHE3PL::normalizeHistogram(const cv::Mat& histogram)
{
  const int n = histogram.cols;
  cv::Mat normalizedHistogram(1, n, CV_64FC1, cv::Scalar(0));

  double sum = 0;
  for(int i = 0; i < n; i++)
    sum += histogram.at<uchar>(0, i);

  for(int i = 0; i < n; i++)
    normalizedHistogram.at<double>(0, i) = (double)histogram.at<uchar>(0, i) / sum;

  return normalizedHistogram;
}

cv::Mat BHE3PL::calculateCummulative(const cv::Mat& histogram)
{
  cv::Mat cummulative = histogram.clone();
  for(int i = 1; i < cummulative.cols; i++)
    cummulative.at<double>(0, i) += cummulative.at<double>(0, i-1);

  return cummulative;
}

double BHE3PL::calculateSeparatingPoint(const cv::Mat& normalizedHistogram)
{  
  double separatingPoint = 0;
  for(int i = 0; i < normalizedHistogram.cols; i++)
    separatingPoint += i*normalizedHistogram.at<double>(0, i);

  return separatingPoint;
}

void BHE3PL::buildSubHistograms()
{
  const int separatingPoint = std::round(_globalSeparatingPoint);

  _subHistogramLow = cv::Mat(1, separatingPoint + 1, CV_8UC1, cv::Scalar(0));
  _subHistogramHigh = cv::Mat(1, _numberOfBins - separatingPoint-1, CV_8UC1, cv::Scalar(0));

  for(int i = 0; i < separatingPoint + 1; i++)
    _subHistogramLow.at<uchar>(0, i) = _globalHistogram.at<uchar>(0, i);

  for(int i = separatingPoint + 1; i < _globalHistogram.cols; i++)
    _subHistogramHigh.at<uchar>(0, i - (separatingPoint+1) ) = _globalHistogram.at<uchar>(0, i);
}

void BHE3PL::applyHistogramSegmentation()
{
  _normalizedGlobalHistogram = normalizeHistogram(_globalHistogram);

  _globalSeparatingPoint = calculateSeparatingPoint(_normalizedGlobalHistogram);

  buildSubHistograms();
}

void BHE3PL::calculateSubHistogramSeparatingPoints()
{
  cv::Mat normalizedSubHistogramLow = normalizeHistogram(_subHistogramLow);
  cv::Mat normalizedSubHistogramHigh = normalizeHistogram(_subHistogramHigh);

  _lowSeparatingPoint = calculateSeparatingPoint(normalizedSubHistogramLow);
  _highSeparatingPoint = _globalSeparatingPoint +
    calculateSeparatingPoint(normalizedSubHistogramHigh);
}

void BHE3PL::calculateGrayLevels1And3(const double grayLevel2, double* grayLevel1, double* grayLevel3)
{
  double grayLevelRatioDifference;
  if( grayLevel2 > 0.5 )    
      grayLevelRatioDifference = (1-grayLevel2) / 2;    
  else
      grayLevelRatioDifference = grayLevel2 / 2;

  *grayLevel1 = grayLevel2 - grayLevelRatioDifference;
  *grayLevel3 = grayLevel2 + grayLevelRatioDifference;
}

void BHE3PL::calculateGrayLevelRatios()
{
  int minimumGrayLevel = 0;
  while(_globalHistogram.at<uchar>(0, minimumGrayLevel) == 0)
    minimumGrayLevel++;

  int maximumGrayLevel = _globalHistogram.cols-1;
  while(_globalHistogram.at<uchar>(0, maximumGrayLevel) == 0)
    maximumGrayLevel--;

  const double lowGrayLevel2 = (_globalSeparatingPoint - _lowSeparatingPoint) / (_globalSeparatingPoint - minimumGrayLevel);
  double lowGrayLevel1, lowGrayLevel3;
  calculateGrayLevels1And3(lowGrayLevel2, &lowGrayLevel1, &lowGrayLevel3);
  
  const double highGrayLevel2 = (maximumGrayLevel - _highSeparatingPoint) / (maximumGrayLevel - _globalSeparatingPoint);
  double highGrayLevel1, highGrayLevel3;
  calculateGrayLevels1And3(highGrayLevel2, &highGrayLevel1, &highGrayLevel3);

  _lowGrayLevelRatios.push_back(lowGrayLevel1);
  _lowGrayLevelRatios.push_back(lowGrayLevel2);
  _lowGrayLevelRatios.push_back(lowGrayLevel3);

  _highGrayLevelRatios.push_back(highGrayLevel1);
  _highGrayLevelRatios.push_back(highGrayLevel2);
  _highGrayLevelRatios.push_back(highGrayLevel3);

  for(const auto& grayLevel: _lowGrayLevelRatios)
    assert(0 <= grayLevel && grayLevel <= 1);
  for(const auto& grayLevel: _highGrayLevelRatios)
  assert(0 <= grayLevel && grayLevel <= 1);
}

void BHE3PL::calculatePlateauLimits()
{
  double lowPeak = 0;
  cv::MatIterator_<uchar> lowIter = _subHistogramLow.begin<uchar>();
  while(lowIter != _subHistogramLow.end<uchar>())
    lowPeak = std::max(lowPeak, (double)*lowIter), ++lowIter;

  double highPeak = 0;
  cv::MatIterator_<uchar> highIter = _subHistogramHigh.begin<uchar>();
  while(highIter != _subHistogramHigh.end<uchar>())
    highPeak = std::max(highPeak, (double)*highIter), ++highIter;

  for(const auto& grayLevelRatio: _lowGrayLevelRatios)
    _lowPlateauLimits.push_back(grayLevelRatio * lowPeak);

  for(const auto& grayLevelRatio: _highGrayLevelRatios)
    _highPlateauLimits.push_back(grayLevelRatio * highPeak);
}

void BHE3PL::calculateModifiedHistogram(const cv::Mat& subHistogram, const VectorDouble& plateauLimits, cv::Mat& modifiedHistogram)
{
  modifiedHistogram = subHistogram.clone();

  cv::MatIterator_<uchar> iter = modifiedHistogram.begin<uchar>();
  while(iter != modifiedHistogram.end<uchar>())
    {
      if( *iter > plateauLimits.at(2) )
	*iter = std::round(plateauLimits.at(2));
      else if( *iter < plateauLimits.at(0) )
	*iter = std::round(plateauLimits.at(0));
      else
	*iter = std::round(plateauLimits.at(1));
      ++iter;
    }  
}

void BHE3PL::applyHistogramModification()
{
  calculateSubHistogramSeparatingPoints();
  calculateGrayLevelRatios();
  calculatePlateauLimits();

  calculateModifiedHistogram(_subHistogramLow, _lowPlateauLimits, _modifiedLowSubHistogram);
  calculateModifiedHistogram(_subHistogramHigh, _highPlateauLimits, _modifiedHighSubHistogram);

  cv::hconcat(_modifiedLowSubHistogram, _modifiedHighSubHistogram, _modifiedGlobalHistogram);
}

void BHE3PL::applyHistogramTransformation()
{
  const cv::Mat normalizedModLowSubHistogram = normalizeHistogram(_modifiedLowSubHistogram);
  const cv::Mat cummulativeModLow = calculateCummulative(normalizedModLowSubHistogram);
  const cv::Mat normalizedModHighSubHistogram = normalizeHistogram(_modifiedHighSubHistogram);
  const cv::Mat cummulativeModHigh = calculateCummulative(normalizedModHighSubHistogram);

  std::cout << "norm low" << normalizedModLowSubHistogram << "\n";
  std::cout << cummulativeModLow << "\n";
  std::cout << cummulativeModHigh << "\n";

  _finalTransformation = cv::Mat(1, _globalHistogram.cols, CV_8UC1, cv::Scalar(0));

  for(int i = 0; i < _finalTransformation.cols; i++)
    {
      if( i < _globalSeparatingPoint )
	{
	  const double value = _globalSeparatingPoint * cummulativeModLow.at<uchar>(i);
	  _finalTransformation.at<uchar>(0, i) = value;
	}
      else
	{
	  double value =
	    (255-_globalSeparatingPoint-1)*(cummulativeModHigh.at<uchar>(i-_globalSeparatingPoint));
	  value += _globalSeparatingPoint + 1;
	  _finalTransformation.at<uchar>(0, i) = value;
	}
    }

  std::cout << "ftrans " << _finalTransformation << "\n";
  
}

#endif /* BHE3PL_H */
