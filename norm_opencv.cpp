#include <opencv2\opencv.hpp>
using namespace cv;

int main(Mat1b img, Mat1b templ)
{
    // Compute match
    Mat result;
    matchTemplate(img, templ, result, TM_CCORR_NORMED);

    // Get best match
    Point maxLoc;
    double maxVal;
    minMaxLoc(result, NULL, &maxVal, NULL, &maxLoc);

    return 0;
}
