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
			//std::cout << "l = " << l << "\n";;
			//std::cout << "X[" << l << "][" << 0 << "] = " << X[l][0] << " ";
			for (int j = 1; j < (layers[l - 1] + 1); j++) {
				int y = 0;
				if (l == 1)
					y = inputCountPerSample + 1;
				else
					y = layers[l - 2] + 1;
				double res = 0.0;
				//std::cout << "\tj = " << j << " y = " << y << "\n";;
				for (int i = 0; i < y; i++) {
					//std::cout << "\t\ti = " << i << "\n";;
					res += W[l][i][j] * X[l - 1][i];
				}
				X[l][j] = std::tanh(res);
				//std::cout << "X[" << l << "][" << j << "] = " << X[l][j] << " ";
			}
			//std::cout << "\n";
		}

	}

	void update_delta(double*** W, double** X, int* layers, int layer_count, double** delta, int inputCountPerSample) {

		for (int l = layer_count; l > 0; l--) {
			int y = 0;
			if (l == 1)
				y = inputCountPerSample + 1;
			else
				y = layers[l - 2] + 1;
			for (int i = 0; i < y; i++) {
				double res = 0.0;
				for (int j = 1; j < (layers[l - 1] + 1); j++) {
					res += W[l][i][j] * delta[l][j];
					std::cout << "res delta[" << l << "][" << j << "] : " << delta[l][j] << "\n\n";
				}
				delta[l - 1][i] = res;
				std::cout << "delta[" << l - 1 << "][" << i << "] : " << delta[l - 1][i] << "\n";
			}
		}

	}

	void update_W(double*** W, double** X, int* layers, int layer_count, int inputCountPerSample, double** delta, int alpha) {
		for (int l = 1; l < layer_count + 1; l++) {
			int y = 0;
			if (l == 1)
				y = inputCountPerSample + 1;
			else
				y = layers[l - 2] + 1;
			for (int i = 0; i < y; i++) {
				for (int j = 1; j < (layers[l - 1] + 1); j++) {
					W[l][i][j] = W[l][i][j] - alpha * (X[l - 1][i] * delta[l][j]);
					std::cout << "W[l][i][j] = " << W[l][i][j] << "\n";
					std::cout << alpha << " * " << X[l - 1][i] << " * " << delta[l][j] << "\n";
				}
				std::cout << "\n";
			}
			std::cout << "\n";
		}
	}

	void get_last_delta(double** X, int* layers, int layer_count, int* Y, double** delta) {

		int L = layer_count;
		for (int j = 1; j < (layers[L]) + 1; j++) {
			delta[L][j] = (1 - std::pow(X[L][j], 2)) * (X[L][j] * Y[j - 1]);
			std::cout << "get_last_delta delta[" << L << "][" << j << "] : " << delta[L][j] << "\n";
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


		double** X = new double* [layer_count + 1];
		int* D = new int[layer_count];

		for (int l = 0; l < layer_count; l++)
			D[l] = layers[l];

		for (int l = 0; l < layer_count + 1; l++)
		{
			//std::cout << "l : " << l << " ";
			if (l == 0)
				X[l] = new double[inputCountPerSample + 1];
			else
				X[l] = new double[layers[l - 1] + 1];
			X[l][0] = 1;
		}
		//std::cout << "\n";

		for (int e = 0; e < epochs; e++)
		{
			int position = 0;
			int posY = 0;
			for (int img = 0; img < sampleCount; img++)
			{
				// Cahrger Xtrain => X[0]

				for (int n = 1; n < (inputCountPerSample + 1); n++)
					X[0][n] = Xtrain[position++];


				feedForward(W, layers, layer_count, inputCountPerSample, X);
				// Charger Y par rapport a YTrain de limage

				int* y = new int[layers[layer_count - 1]];
				double** delta = new double* [layer_count + 1];
				for (int l = 0; l < layer_count + 1; l++)
				{
					if (l == 0)
						delta[l] = new double[inputCountPerSample + 1];
					else
						delta[l] = new double[layers[l - 1] + 1];
				}

				for (int subimg = 0; subimg < layers[layer_count - 1]; subimg++)
				{
					y[subimg] = YTrain[posY++];
					std::cout << y[subimg] << " ";
				}

				get_last_delta(X, layers, layer_count, y, delta);
				update_delta(W, X, layers, layer_count, delta, inputCountPerSample);
				//update_W(W, X, layers, layer_count, inputCountPerSample, delta, alpha);
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
					W[l][i][j] = distribution(generator);
					//W[l][i][j] = k++;
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