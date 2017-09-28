#ifndef T1_TRAINING_H
#define T1_TRAINING_H
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;

class Training {
public:
    Training(String positiveImgs, String negativeImgs, int boxWidth, int boxHeight);
    void train(void);
    void saveDescriptor(String f);
    void setPadding(int x, int y);
    void setStride(int x, int y);

private:
    void calcHOG(Size boxSize, vector<Mat> imgs, vector<Mat> &gradients);
    vector<float> calcDetector(Ptr<SVM> svm);
    String positiveImgs;
    String negativeImgs;
    int boxWidth;
    int boxHeight;
    int strideX = 8;
    int strideY = 8;
    int paddingX = 32;
    int paddingY = 32;
    HOGDescriptor tHOG;
};


#endif
