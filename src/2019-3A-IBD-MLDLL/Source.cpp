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
#include "EnumGame.h"


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
				for (int n = 1; pos < inputCountPerSample * k + inputCountPerSample; pos++, n++)
					nn->Layers[0]->neurons[0]->inputs[n] = XTrain[pos];
				nn->Layers[0]->neurons[0]->inputs[0] = 1;

				feedForwadAll(nn);
				for (int n = 0; n < nn->Layers[0]->neurons[0]->nbInputs; n++)
				{
					nn->Layers[0]->neurons[0]->weights[n] = nn->Layers[0]->neurons[0]->weights[n] +
						alpha * (YTrain[k] - nn->Layers[0]->neurons[0]->output) * nn->Layers[0]->neurons[0]->inputs[n];
					//std::cout << "update w[" << n << "] " << nn->Layers[0]->neurons[0]->weights[n] << "\n";
					// W = W + a(Yk - g(Xk)) + Xk
				}
				//printNN(nn);
				Xout[k] = nn->Layers[0]->neurons[0]->output;


			}

			double loss = mse_loss(YTrain, Xout, sampleCount);
			if (e % 10 == 0 || e == epochs - 1)
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
		return nn->Layers[0]->neurons[0]->output;
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
		int h = 200;
		int inputCountPerSample = w * h;
		double alpha = 0.001;
		int epochs = 100;
		auto class_ = FPS;
		std::cout << "Please wait until we load " << nbImages * 3 << " images of size " << w << "x" << h << "\n";
		double* XTrain = buildXTrain("../../img/FPS/", "../../img/RTS/", "../../img/MOBA/", w, h, nbImages);

		double* YTrain = buildYTrain(nbImages, class_);

		// Build
		auto W = create_linear_model(inputCountPerSample);

		// Fit
		W = fit_classification_rosenblatt_rule(W, XTrain, sampleCount, inputCountPerSample, YTrain, alpha, epochs);

		// Prediction
		double* XPredict = loadImgToPredict("../../img/MOBA_Test/", w, h);
		auto prediction = predict_classification(W, XPredict, inputCountPerSample);

		if (prediction >= 1)
			std::cout << "I think this image is from class: " << getGame(class_) << "\n";
		else
			std::cout << "I don't think this image is from class: " << getGame(class_) << "\n";

		std::cin.get();
		return 0;
	}
}