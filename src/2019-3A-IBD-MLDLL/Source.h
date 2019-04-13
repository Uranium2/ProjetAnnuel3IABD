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
#include "NeuralNet.h"

SUPEREXPORT double* create_linear_model(int inputCountPerSample);

SUPEREXPORT void fit_classification_rosenblatt_rule(
	double* W,
	double* XTrain,
	int sampleCount,
	int inputCountPerSample,
	double* YTrain,
	double alpha, // Learning Rate
	int epochs // Nombre d'itération
);