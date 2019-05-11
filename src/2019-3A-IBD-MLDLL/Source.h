#pragma once
#if _WIN32
#define SUPEREXPORT __declspec(dllexport)
#else
#define SUPEREXPORT 
#endif

#include <random>
#include <iostream>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

SUPEREXPORT double* create_linear_model(int inputCountPerSample);
SUPEREXPORT double* test_python_array(double* W, int inputCountPerSample);

SUPEREXPORT void fit_classification_rosenblatt_rule(
	double* W,
	double* XTrain,
	int sampleCount,
	int inputCountPerSample,
	double* YTrain,
	double alpha,
	int epochs
);
SUPEREXPORT double predict_regression(double* W, double* X, int inputCountPerSample);