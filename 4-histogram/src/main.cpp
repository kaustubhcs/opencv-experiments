#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <fstream> // For logging data
using namespace cv;
using namespace std;

extern void transpose_img(unsigned char *in_mat,
                          unsigned int height, 
                          unsigned int width,
                          unsigned long long int *hist,
                          int low_thresh,
                          int high_thresh);

double CLOCK() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC,  &t);
    return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}
int low_thresh=0, high_thresh=255;
long debugger=0;

int main( int argc, const char** argv ) { 
        // arg 1: Input image 

    switch (argc) { // For getting input from console
        case 5:

            int input_5;
            input_5 = atoi(argv[4]); //Fourth Input
            debugger = input_5;

        case 4:
            int input_4;

            input_4 = atoi(argv[3]); //Third Input
            high_thresh = input_4;

        case 3:
            int input_3;
            input_3 = atoi(argv[2]); // Second Input
            low_thresh = input_3;

        case 2:
            /*
            long input_1;
            input_1 = atol(argv[1]); // First input
            mat_size = input_1;
             */
            break;
        case 1:
            // Keep this empty
            break;
        default:
            cout << "FATAL ERROR: Wrong Number of Inputs" << endl; // If incorrect number of inputs are used.
            return 1;
    }


        
        double start_gpu, finish_gpu;
        unsigned long long int *hist;
        hist = new unsigned long long int [256];
        
        // Read input image from argument in black and white
        Mat input_image = imread(argv[1], IMREAD_GRAYSCALE);

        if (input_image.empty()){
            cout << "Image cannot be loaded..!!" << endl;
            return -1;
        }
        
        unsigned int height = input_image.rows;
        unsigned int  width = input_image.cols;
        
        //////////////////////////
        // START GPU Processing //
        //////////////////////////
        
        start_gpu = CLOCK();
   
        // New mat has inverted height/width
        Mat transpose = Mat::zeros(height, width, CV_8U);


        transpose_img((unsigned char *) input_image.data,
                               height, 
                               width,
                               hist,
                               low_thresh,
                               high_thresh);

        finish_gpu = CLOCK();

    unsigned long long int tot=0;
    for (int i=0;i<256;i++) {
        tot += hist[i];
       //printf("%d %c ",i,i);
        //cout << "H["<<i<<"]:" << hist[i] << "\t";
    }
    cout << endl;
    cout << "TOTAL PIXELS CALCULATED: " << tot << endl;
        
        cout << "GPU execution time: " << finish_gpu - start_gpu << " ms" << endl;

    cout << "\n\nHISTOGRAM...\n\n\n";

    for (int i=low_thresh;i<=high_thresh;i++) {
        cout << "Pixel Value: " << i << "\t";
        int cval = (int)hist[i]/1500.0;
        for (int j=0;j<cval;j++) {
            if (cval < 150) {
                cout << "\033[1;32m";
            } else {
                cout << "\033[1;31m";
            }
            cout << "*";
            cout << "\033[0m";
        }
        cout << endl;
    }

    // Writing data to file
        ofstream log_m;
    log_m.open("data_log.csv", ios::app);
    log_m << "Pixel, ";
    log_m <<"Frequency" << endl;


    for (int i=0;i < 256;i++) {
        log_m << i << ",";
        log_m << hist[i];
        log_m << endl;
    }
    log_m.close();




        return 0;
}
