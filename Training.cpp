#include "Training.h"
#include "Utility.h"

Training::Training(String positiveImgs, String negativeImgs, int boxWidth, int boxHeight){
    this->positiveImgs = positiveImgs;
    this->negativeImgs = negativeImgs;
    this->boxHeight = boxHeight;
    this->boxWidth = boxWidth;
}

void Training::setPadding(int x, int y){
    paddingX = x;
    paddingY = y;
}

void Training::setStride(int x, int y){
    strideX = x;
    strideY = y;
}

void Training::calcHOG(Size boxSize, vector<Mat> imgs, vector<Mat> &gradients){
    HOGDescriptor hogDescriptor;
    hogDescriptor.winSize = boxSize;

    Rect rect = Rect(0, 0, boxSize.width, boxSize.height);
    Mat gray;
    vector<float> descVecs;

    for(int i=0 ; i < imgs.size(); i++){
        hogDescriptor.compute(imgs[i](rect), descVecs, Size(strideX, strideX), Size(paddingX, paddingX));
        gradients.push_back(Mat(descVecs));
    }
}

vector<float> Training::calcDetector(Ptr<SVM> svm){
    Mat sv = svm->getSupportVectors();
    vector<float> detector(sv.cols + 1);

    Mat alpha, i;
    double rho = svm->getDecisionFunction(0, alpha, i);
    memcpy(&detector[0], sv.ptr(), sv.cols*sizeof(float));
    detector[sv.cols] = (float)-rho;
    return detector;
}

void Training::saveDescriptor(String f){
    tHOG.save(f);
}

void Training::train(){
    vector<Mat> posSet, negSet, negSetTrain, gradients;
    Mat trainData;
    vector<float> hogDetector;
    vector<int> numSamples;

    Utility::loadImgs(positiveImgs, posSet);
    Utility::loadImgs(negativeImgs, negSet);
    Size boxSize = Size(boxWidth, boxHeight);
    tHOG.winSize = boxSize;
    negSetTrain = Utility::cropImgRandom(negSet, boxSize);

    for(int i = 0; i < posSet.size();i++)
        numSamples.push_back(1);
    for(int i = 0;i < negSetTrain.size();i++)
        numSamples.push_back(-1);

    calcHOG(boxSize, posSet, gradients);
    calcHOG(boxSize, negSetTrain, gradients);

    Utility::convertData(gradients, trainData);

    Ptr<SVM> svm = SVM::create();
    svm->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, 1e-3));
    svm->setKernel(SVM::LINEAR);
    svm->setNu(0.5);
    svm->setP(0.1);
    svm->setC(0.01);
    svm->setType(SVM::EPS_SVR);
    svm->train(trainData, ROW_SAMPLE, Mat(numSamples));

    hogDetector = calcDetector(svm);
    tHOG.setSVMDetector(hogDetector);
}