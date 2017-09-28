#ifndef T1_UTILITY_H
#define T1_UTILITY_H

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class Utility {
public:
    static void loadImgs(String dir, vector<Mat> &imgs);
    static vector<Mat> cropImgRandom(vector<Mat> negSet, Size size);
    static void convertData( const vector< Mat > & train_samples, Mat& train_matrix );
};


#endif
