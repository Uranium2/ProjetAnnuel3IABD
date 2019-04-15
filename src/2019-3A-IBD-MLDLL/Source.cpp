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
		auto W = new double[inputCountPerSample + 1];
		double low = -1.0;
		double up = 1.0;
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::default_random_engine generator(seed);

		std::uniform_real_distribution<double> distribution(low, up);
		for (int i = 0; i < inputCountPerSample + 1; i++)
		{
			W[i] = distribution(generator);
		}
		// TODO : initialisation random [-1,1]
		return W;
	}

	SUPEREXPORT void fit_classification_rosenblatt_rule(
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
		NeuralNet* nn = buildNeuralNet(W, nbLayers, sizeLayers);
		nn->inputs = XTrain;
		for (auto i = 0; i < epochs; i++)
		{
			double* Xout = new double[sampleCount];
			for (auto k = 0; k < sampleCount; k++)
			{
				for (int n = 0; n < nn->Layers[0]->nbNeurons; n++)
				{
					nn->Layers[0]->neurons[0]->weights[0] = nn->Layers[0]->neurons[n]->weights[0] + alpha * (YTrain[k] - nn->Layers[0]->neurons[n]->output) * nn->inputs[k];
					std::cout << "weight for neuron n = " << n << ": " << nn->Layers[0]->neurons[0]->weights[0] << "\n";
					// W = W + a(Yk - g(Xk)) + Xk
				}
				Xout[k] = nn->Layers[nbLayers - 1]->neurons[0]->output;

				
				feedForwadAll(nn);
			}
			double loss = mse_loss(YTrain, Xout, 4);
			printf("Epoch: %d loss: %f\n", i, loss);
			printNN(nn);
			
		}
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
		return predict_regression(W, XToPredict, inputCountPerSample) >= 0 ? 1.0 : -1.0;
	}

	SUPEREXPORT void delete_linear_model(double* W)
	{
		delete[] W;
	}

	int main()
	{
		int sampleCount = 1;
		int inputCountPerSample = 1;
		int nbImages = 1;
		int w = 1;
		int h = 2;
		double* XTrain = buildXTrain("../../img/A/", "../../img/B/", "../../img/C/", w, h, nbImages);
		double* YTrain = buildYTrain(nbImages, 1);
		double alpha = 0.05;
		int epochs = 10;
		auto W = create_linear_model(inputCountPerSample);

		fit_classification_rosenblatt_rule(W, XTrain, sampleCount, inputCountPerSample, YTrain, alpha, epochs);

		std::cin.get();
		return 0;
	}
}