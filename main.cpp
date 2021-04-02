#include <opencv2/highgui.hpp>
#include "cnpy.h"
#include <complex>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

int main () {
    cv::Mat image;
    cnpy::NpyArray color_array = cnpy::npy_load("test.npy");
    image = cv::imread("./test.jpeg", cv::IMREAD_COLOR);
    cv::imshow("Image", image);
    cv::waitKey(0);
    return 0;
}