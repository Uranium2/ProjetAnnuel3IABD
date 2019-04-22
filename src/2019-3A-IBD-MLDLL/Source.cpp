#if _WIN32
#define SUPEREXPORT __declspec(dllexport)
#else
#define SUPEREXPORT 
#endif
#define _SILENCE_CXX17_NEGATORS_DEPRECATION_WARNING

#include <random>
#include <iostream>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "NeuralNet.h"
#include "ImgToArr.h"


extern "C" {


	SUPEREXPORT double* create_linear_model(int inputCountPerSample)
	{
		auto W = new double[(double)inputCountPerSample];
		double low = -1.0;
		double up = 1.0;
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::default_random_engine generator(seed);

		std::uniform_real_distribution<double> distribution(low, up);
		for (int i = 0; i < inputCountPerSample; i++)
		{
			W[i] = distribution(generator);
		}
		// TODO : initialisation random [-1,1]
		return W;
	}

	SUPEREXPORT double* fit_classification_rosenblatt_rule(
		double* W,
		double* XTrain,
		int sampleCount,
		int inputCountPerSample,
		double* YTrain,
		double alpha, // Learning Rate
		int epochs // Nombre d'itération
	)
	{
		int* sizeLayers = new int[1];
		sizeLayers[0] = 1;
		int nbLayers = 1;
		NeuralNet* nn = buildNeuralNet(W, nbLayers, sizeLayers, inputCountPerSample);

		for (int i = 0; i < inputCountPerSample; i++)
			nn->Layers[0]->neurons[0]->weights[i] = W[i];

		nn->Layers[0]->neurons[0]->nbInputs = inputCountPerSample;


		for (int e = 0; e < epochs; e++) {
			int pos = 0;
			double* Xout = new double[(double)sampleCount];
			for (int k = 0; k < sampleCount; k++)
			{
				for (int n = 0; pos < inputCountPerSample * k + inputCountPerSample; pos++, n++)
					nn->Layers[0]->neurons[0]->inputs[n] = XTrain[pos];

				feedForwadAll(nn);
				for (int n = 0; n < nn->Layers[0]->neurons[0]->nbInputs; n++)
				{
					nn->Layers[0]->neurons[0]->weights[n] = nn->Layers[0]->neurons[0]->weights[n] + alpha * (YTrain[k] - nn->Layers[0]->neurons[0]->output) * nn->Layers[0]->neurons[0]->inputs[n];
					//std::cout << "update w[" << n << "] " << nn->Layers[0]->neurons[0]->weights[n] << "\n";
					// W = W + a(Yk - g(Xk)) + Xk
				}
				//printNN(nn);
				Xout[k] = nn->Layers[0]->neurons[0]->output;


			}

			double loss = mse_loss(YTrain, Xout, sampleCount);
			printf("Epoch: %d loss: %f\n", e, loss);

		}
		return nn->Layers[0]->neurons[0]->weights;
	}



	SUPEREXPORT void fit_regression(
		double* W,
		double* XTrain,
		int SampleCount,
		int inputCountPerSample,
		double* YTrain
	)
	{
		// TODO : entrainement (correction des W, cf slides !)
	}

	SUPEREXPORT double predict_regression(
		double* W,
		double* XToPredict,
		int inputCountPerSample
	)
	{
		// TODO : Inférence (CF Slides !)
		return 0.42;
	}

	SUPEREXPORT double predict_classification(
		double* W,
		double* XToPredict,
		int inputCountPerSample
	)
	{
		int* sizeLayers = new int[1];
		sizeLayers[0] = 1;
		int nbLayers = 1;
		NeuralNet* nn = buildNeuralNet(W, nbLayers, sizeLayers, inputCountPerSample);

		for (int i = 0; i < inputCountPerSample; i++)
		{
			nn->Layers[0]->neurons[0]->weights[i] = W[i];
			nn->Layers[0]->neurons[0]->inputs[i] = XToPredict[i];
		}
		feedForwadAll(nn);
		std::cout << nn->Layers[0]->neurons[0]->output;
		return nn->Layers[0]->neurons[0]->output >= 0 ? 1.0 : -1.0;
		//return predict_regression(W, XToPredict, inputCountPerSample) >= 0 ? 1.0 : -1.0;
	}

	SUPEREXPORT void delete_linear_model(double* W)
	{
		delete[] W;
	}

	int main()
	{
		// Build param
		int nbImages = 10;
		int sampleCount = nbImages * 3;
		int w = 100;
		int h = 23;
		int inputCountPerSample = w * h;
		double alpha = 0.05;
		int epochs = 100;

		double* XTrain = buildXTrain("../../img/FPS/", "../../img/RTS/", "../../img/MOBA/", w, h, nbImages);
		double* YTrain = buildYTrain(nbImages, 3);

		// Build
		auto W = create_linear_model(inputCountPerSample);

		// Fit
		W = fit_classification_rosenblatt_rule(W, XTrain, sampleCount, inputCountPerSample, YTrain, alpha, epochs);

		// Prediction
		double* XPredict = buildXTrain("../../img/FPS_Predict/", "../../img/RTS_Predict/", "../../img/MOBA_Predict/", w, h, nbImages);
		std::cout << predict_classification(W, XPredict, inputCountPerSample) << "\n";

		std::cin.get();
		return 0;
	}
}