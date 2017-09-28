#include "Utility.h"


void Utility::loadImgs(String dir, vector<Mat> &imgs){
    vector<String> f;

    glob(dir, f);
    for(int i = 0; i < f.size(); i++){
        Mat img = imread(f[i]);
        if (img.empty())
            continue;
        imgs.push_back(img);
    }
}

vector<Mat> Utility::cropImgRandom(vector<Mat> negSet, Size size){
    Rect rect;
    rect.width = size.width;
    rect.height = size.height;
    vector<Mat> cropSet;
    srand((unsigned int)time(NULL));
    int x = rect.width;
    int y = rect.height;

    for(int i = 0; i < negSet.size(); i++){
        rect.x = rand() % (negSet[i].cols-x);
        rect.y = rand() % (negSet[i].rows-y);
        Mat neg = negSet[i](rect);
        cropSet.push_back(neg);
    }
    return cropSet;
}


//OpenCV Function to convert data to be used on ML algorithms
void Utility::convertData( const vector< Mat > & train_samples, Mat& train_matrix ){
    int rows = train_samples.size();
    int cols = (int)std::max( train_samples[0].cols, train_samples[0].rows );
    Mat tmp( 1, cols, CV_32FC1 );
    train_matrix = Mat( rows, cols, CV_32FC1 );

    for(int i = 0 ; i < train_samples.size(); i++){
        if( train_samples[i].cols == 1 )
        {
            transpose( train_samples[i], tmp );
            tmp.copyTo( train_matrix.row( i ));
        }
        else if( train_samples[i].rows == 1 )
        {
            train_samples[i].copyTo( train_matrix.row( i ));
        }
    }
}