#pragma once

typedef struct mlp
{
	int* layers;
	int layer_count;
	double*** W;
	double* XTrain;
	double* YTrain;
	double** Xall;
	double** Yall;
	double** delta;
	double* Y;
	double alpha;
	int epochs;
} MLP;

void printMLP(double*** W, int* layers, int layer_count, int inputCountPerSample);
double*** create_mlp_model(int* layers, int layer_count, int inputCountPerSample);
MLP* build_mlp(double*** W, int* layers, int layer_count, double* XTrain, double* YTrain, int sampleCount, int inputCountPerSample, double alpha, int epochs);
double*** fit_mlp_classification(double*** W,
	double* XTrain,
	double* YTrain,
	int* layers,
	int layer_count,
	int sampleCount,
	int inputCountPerSample,
	double alpha,
	int epochs);
