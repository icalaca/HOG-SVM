#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>
#include <GL/glut.h>
#include "Utility.h"
#include "Training.h"


using namespace cv;
using namespace cv::ml;
using namespace std;

vector<Mat> myImgs;
HOGDescriptor myHog;
vector<Rect> rects;
vector<double> hogWeights;
int imgindex = 0;
int winH = 500;
int winW = 600;


void loadTex(Mat img){
    glClearColor (0.0, 0.0, 0.0, 0.0);
    Mat tImg;
    flip(img,tImg,0);
    gluBuild2DMipmaps(GL_TEXTURE_2D, 3,tImg.cols,tImg.rows,
                      GL_BGR, GL_UNSIGNED_BYTE, tImg.data);
}

void resizeWindow(int w, int h){
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, winW, 0, winH, -1, 1);
}


void drawScene(void){
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_TEXTURE_2D);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0);
    glVertex3f(0, 0, 0.0);

    glTexCoord2f(0.0, 1.0);
    glVertex3f(0, winH, 0.0);

    glTexCoord2f(1.0, 1.0);
    glVertex3f(winW, winH, 0.0);

    glTexCoord2f(1.0, 0.0);
    glVertex3f(winW, 0, 0.0);
    glEnd();
    glFlush();
    glDisable(GL_TEXTURE_2D);

    if(imgindex >= myImgs.size())
        return;

    Mat img = myImgs[imgindex];
    myHog.detectMultiScale(img, rects, hogWeights);
    for(int i = 0; i < rects.size(); i++){
        if(hogWeights[i] >= 0.85){
            Rect r = rects[i];
            Scalar color = Scalar( 0, 230, 0 );
            rectangle( img, r, color, 2 );
        }
    }
    loadTex(img);
    imgindex++;
    glutPostRedisplay();
}

int main( int argc, char** argv ){
    bool training = false;
    String posImgs = "./cn03dataset";
    String negImgs = "./dataset/NonPedestrians";
    String imgs = "./CN-03";
    String trainFile = "myhogcn03.yml";

    if(training){
        Training t(posImgs, negImgs, 48, 96);
        t.setPadding(32, 32);
        t.setStride(8, 8);
        t.train();
        t.saveDescriptor(trainFile);
    }

    Utility::loadImgs(imgs, myImgs);
    myHog.load(trainFile);
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(winW, winH);
    glutInitWindowPosition(100, 100);
    glutCreateWindow(":D");
    glutDisplayFunc(drawScene);
    glutReshapeFunc(resizeWindow);
    glutMainLoop();
    return 0;
}