#if _WIN32
#define SUPEREXPORT __declspec(dllexport)
#else
#define SUPEREXPORT 
#endif
#include <random>
#include <chrono>
#include <iostream>
extern "C" {

	SUPEREXPORT void printArrayPython3D(double*** W, int* layers, int layer_count, int inputCountPerSample)
	{
		for (int l = 1; l < (layer_count + 1); l++) {
			int y = 0;
			if (l == 1)
				y = inputCountPerSample + 1;
			else
				y = layers[l - 2] + 1;
			for (int i = 0; i < y; i++) {
				for (int j = 1; j < (layers[l - 1] + 1); j++) {
					std::cout << W[l][i][j] << " ";
				}
				std::cout << "\n";
			}
			std::cout << "\n";
		}
	}

	void feedForward(double*** W, int* layers, int layer_count, int inputCountPerSample, double** X) {

		for (int l = 1; l < (layer_count + 1); l++) {
			std::cout << "l = " << l << "\n";;
			for (int j = 1; j < (layers[l - 1] + 1); j++) {
				int y = 0;
				if (l == 1)
					y = inputCountPerSample + 1;
				else
					y = layers[l - 2] + 1;
				double res = 0.0;
				std::cout << "\tj = " << j << " y = " << y << "\n";;
				for (int i = 0; i < y; i++) {
					std::cout << "\t\ti = " << i << "\n";;
					res += W[l][i][j];
				}
				X[l][j] = std::tanh(res);
				std::cout << "X[l][j] = " << X[l][j] << " ";
			}
			std::cout << "\n";
		}

	}

	SUPEREXPORT void fit_mlp_classification(double*** W,
		double* Xtrain,
		int* YTrain,
		int* layers,
		int layer_count,
		int sampleCount,
		int inputCountPerSample,
		double alpha,
		int epochs) {

		for (int i = 0; i < inputCountPerSample * sampleCount; i++)
		{
			std::cout << Xtrain[i] << " ";
		}
		double** X = new double* [layer_count + 1];
		int* D = new int[layer_count];

		for (int l = 0; l < layer_count; l++)
			D[l] = layers[l];

		for (int l = 0; l < layer_count + 1; l++)
		{
			std::cout << "l : " << l << " ";
			if (l == 0)
				X[l] = new double[inputCountPerSample + 1];
			else
				X[l] = new double[layers[l - 1] + 1];
			X[l][0] = 1;
		}
		std::cout << "\n";

		for (int e = 0; e < epochs; e++)
		{
			int position = 0;
			for (int img = 0; img < sampleCount; img++)
			{
				// Cahrger Xtrain => X[0]
				
				for (int n = 1; n < (inputCountPerSample + 1); n++)
				{
					std::cout << "position = " << position << "\n";
					std::cout << "Xtrain[position] = " << Xtrain[position] << "\n";
					X[0][n] = Xtrain[position++];
					
					//std::cout << "X[0][n] = " << X[0][n] << " ";
				}
				std::cout << "\n";
				//FeedForward (Mise a jour des X)
				//
				// Calcul des delta
				// mise jour des poids W
			}
		}
		
	}

	SUPEREXPORT double*** create_mlp_model(int* layers, int layer_count, int inputCountPerSample)
	{
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::default_random_engine generator(seed);
		std::uniform_real_distribution<double> distribution(-1, 1);

		double*** W = new double** [layer_count + 1];

		int k = 0;
		for (int l = 1; l < (layer_count + 1); l++) {
			int y = 0;
			if (l == 1)
				y = inputCountPerSample + 1;
			else
				y = layers[l - 2] + 1;
			W[l] = new double* [y];
			for (int i = 0; i < y; i++) {
				W[l][i] = new double[layers[l - 1] + 1];
				for (int j = 1; j < (layers[l - 1] + 1); j++) {
					//W[l][i][j] = distribution(generator);
					W[l][i][j] = k++;
				}
			}
		}
		return W;
	}



	int main() {
		int epoch = 1;
		double alpha = 0.01;
		double XTrain[8] = { 0, 0, 0, 1, 1, 0, 1, 1 };
		int YTrain[4] = { -1, -1, -1, 1 };
		int layers[2] = { 2, 1 };
		int layer_count = 2;
		int inputCountPerSample = 2;
		int sampleCount = 4;
		double*** W = create_mlp_model(layers, layer_count, inputCountPerSample);
		fit_mlp_classification(W, XTrain, YTrain, layers, layer_count, sampleCount, inputCountPerSample, alpha, epoch);
		std::cin.get();
		return 0;
	}

}