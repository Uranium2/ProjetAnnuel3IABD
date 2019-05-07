#if _WIN32
#define SUPEREXPORT __declspec(dllexport)
#else
#define SUPEREXPORT 
#endif
#define _SILENCE_CXX17_NEGATORS_DEPRECATION_WARNING

#include <chrono>
#include <random>
#include <algorithm>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/QR>    
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>


extern "C" {
	SUPEREXPORT double* create_linear_model(int inputCountPerSample) {
		double* W = new double[inputCountPerSample + 1];
		double low = -1.0;
		double up = 1.0;
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::default_random_engine generator(seed);

		std::uniform_real_distribution<double> distribution(low, up);
		for (int i = 0; i < inputCountPerSample + 1; i++)
		{
			W[i] = distribution(generator);
		}
		return W;
	}
	double squared_error(double v_true, double v_given)
	{
		return pow(v_true - v_given, 2);
	}

	double mse_loss(double* v_true, double* v_given, int nb_elem)
	{
		double res = 0.0;
		for (int i = 0; i < nb_elem; i++)
		{
			res += squared_error(v_true[i], v_given[i]);
		}
		return res / nb_elem;
	}
	double feedForward(double* X, double* W, int inputCountPerSample)
	{
		double result = 0;
		for (int i = 0; i < inputCountPerSample; i++)
		{
			result += X[i] * W[i];
		}
		return std::tanh(result);
	}

	SUPEREXPORT double predict_regression(double* W, double* X, int inputCountPerSample)
	{
		double* Xnew = new double[inputCountPerSample + 1];
		Xnew[0] = 1;
		int pos = 0;
		for (int i = 1; i < inputCountPerSample + 1; i++)
			Xnew[i] = X[pos++];

		double result = 0;
		for (int i = 0; i < inputCountPerSample + 1; i++)
			result += Xnew[i] * W[i];

		return std::tanh(result);
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
		double output = 0.0;
		double* Xactual = new double[inputCountPerSample + 1];
		Xactual[0] = 1;
		for (int e = 1; e < epochs + 1; e++)
		{
			double* Xout = new double[(double)sampleCount];
			int pos = 0;
			for (int img = 0; img < sampleCount; img++)
			{
				for (int input = 1; input < inputCountPerSample + 1; input++)
					Xactual[input] = XTrain[pos++];

				output = feedForward(Xactual, W, inputCountPerSample + 1);
				Xout[img] = output;

				for (int i = 0; i < inputCountPerSample + 1; i++)
					W[i] = W[i] + alpha * (YTrain[img] - output) * Xactual[i];

			}
			double loss = mse_loss(YTrain, Xout, sampleCount);
			if (e % 10 == 0 || e == epochs - 1)
				printf("Epoch: %d loss: %f\n", e, loss);
		}
	}

	int main()
	{
		double XTrain[8] = { 0,0,
							0,1,
							1,0,
							1,1 };
		int sampleCount = 4;
		int inputCountPerSample = 2;
		double YTrain[4] = { -1, 1, 1, 1 };
		double alpha = 0.01;
		int epochs = 200;
		double* W = create_linear_model(inputCountPerSample);
		fit_classification_rosenblatt_rule(W, XTrain, sampleCount, inputCountPerSample, YTrain, alpha, epochs);

		double input0[2] = { 0, 0 };
		double input1[2] = { 0, 1 };
		double input2[2] = { 1, 0 };
		double input3[2] = { 1, 1 };
		std::cout << predict_regression(W, input0, inputCountPerSample) << "\n";
		std::cout << predict_regression(W, input1, inputCountPerSample) << "\n";
		std::cout << predict_regression(W, input2, inputCountPerSample) << "\n";
		std::cout << predict_regression(W, input3, inputCountPerSample) << "\n";
		std::cin.get();
		return 0;
	}
}