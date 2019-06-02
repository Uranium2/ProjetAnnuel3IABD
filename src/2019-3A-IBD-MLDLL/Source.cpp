#include "Linear.h"
#include "Mlp.h"

int main() {
	int layers[3] = { 2, 2, 1 };
	double Xtrain[8] = { 0, 0,
						0, 1,
						1, 0,
						1, 1 };
	int YTrain[4] = { 1, -1, -1, 1 };
	int sampleCount = 4;
	double alpha = 0.01;
	int epochs = 1000;
	int layer_count = 3;
	int inputCountPerSample = 2;
	auto W = create_mlp_model(layers, layer_count, inputCountPerSample);
	fit_mlp_classification(W, Xtrain, YTrain, layers, layer_count, sampleCount, inputCountPerSample, alpha, epochs);

	double Xpredict0[2] = { 0, 0 };
	auto predict0 = predict_mlp_classification(W, layers, layer_count, inputCountPerSample, Xpredict0);
	std::cout << predict0[1] << "\n";
	double Xpredict1[2] = { 0, 1 };
	auto predict1 = predict_mlp_classification(W, layers, layer_count, inputCountPerSample, Xpredict1);
	std::cout << predict1[1] << "\n";
	double Xpredict2[2] = { 1, 0 };
	auto predict2 = predict_mlp_classification(W, layers, layer_count, inputCountPerSample, Xpredict2);
	std::cout << predict2[1] << "\n";
	double Xpredict3[2] = { 1, 1 };
	auto predict3 = predict_mlp_classification(W, layers, layer_count, inputCountPerSample, Xpredict3);
	std::cout << predict3[1] << "\n";
	return 0;
}