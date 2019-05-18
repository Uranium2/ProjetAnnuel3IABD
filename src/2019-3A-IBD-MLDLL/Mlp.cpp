#include "Mlp.h"

extern "C" {

	double squared_error_mlp(double v_true, double v_given)
	{
		return pow(v_true - v_given, 2);
	}

	double mse_loss_mlp(double* v_true, double* v_given, int nb_elem)
	{
		double res = 0.0;
		for (int i = 0; i < nb_elem; i++)
		{
			res += squared_error_mlp(v_true[i], v_given[i]);
		}
		return res / nb_elem;
	}

	SUPEREXPORT double predict_mlp_classification(double*** W, int* layers, int layer_count, int inputCountPerSample, double* Xinput) {

		double** X = new double* [layer_count + 1];

		for (int l = 0; l < layer_count + 1; l++)
		{
			if (l == 0)
			{
				X[l] = new double[inputCountPerSample + 1];
				int pos = 0;
				for (int input = 1; input < inputCountPerSample + 1; input++)
					X[l][input] = Xinput[pos++];
			}
			else
				X[l] = new double[layers[l - 1] + 1];
			X[l][0] = 1;
		}

		for (int l = 1; l < (layer_count + 1); l++) {
			for (int j = 1; j < (layers[l - 1] + 1); j++) {
				int y = 0;
				if (l == 1)
					y = inputCountPerSample + 1;
				else
					y = layers[l - 2] + 1;
				double res = 0.0;
				for (int i = 0; i < y; i++)
					res += W[l][i][j] * X[l - 1][i];
				X[l][j] = std::tanh(res);
			}
		}
		return X[layer_count][layers[layer_count - 1]];
	}

	SUPEREXPORT double predict_mlp_regression(double*** W, int* layers, int layer_count, int inputCountPerSample, double* Xinput) {

		double** X = new double* [layer_count + 1];

		for (int l = 0; l < layer_count + 1; l++)
		{
			if (l == 0)
			{
				X[l] = new double[inputCountPerSample + 1];
				int pos = 0;
				for (int input = 1; input < inputCountPerSample + 1; input++)
					X[l][input] = Xinput[pos++];
			}
			else
				X[l] = new double[layers[l - 1] + 1];
			X[l][0] = 1;
		}

		for (int l = 1; l < (layer_count + 1); l++) {
			for (int j = 1; j < (layers[l - 1] + 1); j++) {
				int y = 0;
				if (l == 1)
					y = inputCountPerSample + 1;
				else
					y = layers[l - 2] + 1;
				double res = 0.0;
				for (int i = 0; i < y; i++)
					res += W[l][i][j] * X[l - 1][i];
				if (l == layer_count)
					X[l][j] = res;
				else
					X[l][j] = std::tanh(res);
			}
		}
		return X[layer_count][layers[layer_count - 1]];
	}


	void feedForward_mlp_regression(double*** W, int* layers, int layer_count, int inputCountPerSample, double** X) {

		for (int l = 1; l < (layer_count + 1); l++) {
			for (int j = 1; j < (layers[l - 1] + 1); j++) {
				int y = 0;
				if (l == 1)
					y = inputCountPerSample + 1;
				else
					y = layers[l - 2] + 1;
				double res = 0.0;
				for (int i = 0; i < y; i++)
					res += W[l][i][j] * X[l - 1][i];
				if (l == layer_count)
					X[l][j] = res;
				else
					X[l][j] = std::tanh(res);
			}
		}

	}

	void feedForward_mlp(double*** W, int* layers, int layer_count, int inputCountPerSample, double** X) {

		for (int l = 1; l < (layer_count + 1); l++) {
			for (int j = 1; j < (layers[l - 1] + 1); j++) {
				int y = 0;
				if (l == 1)
					y = inputCountPerSample + 1;
				else
					y = layers[l - 2] + 1;
				double res = 0.0;
				for (int i = 0; i < y; i++)
					res += W[l][i][j] * X[l - 1][i];
				X[l][j] = std::tanh(res);
			}
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
				for (int j = 1; j < (layers[l - 1] + 1); j++)
					res += W[l][i][j] * delta[l][j];

				delta[l - 1][i] = (1 - std::pow(X[l - 1][i], 2)) * res;
			}
		}

	}

	void update_W(double*** W, double** X, int* layers, int layer_count, int inputCountPerSample, double** delta, double alpha) {
		for (int l = 1; l < layer_count + 2; l++) {
			int y = 0;
			if (l == 1)
				y = inputCountPerSample + 1;
			else
				y = layers[l - 2] + 1;
			for (int i = 0; i < y; i++)
				for (int j = 1; j < (layers[l - 1] + 1); j++)
					W[l][i][j] = W[l][i][j] - (alpha * X[l - 1][i] * delta[l][j]);
		}
	}

	void get_last_delta(double** X, int* layers, int layer_count, int* Y, double** delta) {

		int L = layer_count;
		for (int j = 1; j < layers[L - 1] + 1; j++) {
			delta[L][j] = (1 - std::pow(X[L][j], 2)) * (X[L][j] - Y[j - 1]);
		}

	}

	void get_last_delta_regression(double** X, int* layers, int layer_count, int* Y, double** delta) {

		int L = layer_count;
		for (int j = 1; j < layers[L - 1] + 1; j++) {
			delta[L][j] = (X[L][j] - Y[j - 1]);
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
			if (l == 0)
				X[l] = new double[inputCountPerSample + 1];
			else
				X[l] = new double[layers[l - 1] + 1];
			X[l][0] = 1;
		}

		for (int e = 0; e < epochs; e++)
		{
			int position = 0;
			int posY = 0;
			double* Xout = new double[sampleCount];
			for (int img = 0; img < sampleCount; img++)
			{
				// Charger Xtrain => X[0]

				for (int n = 1; n < (inputCountPerSample + 1); n++)
					X[0][n] = Xtrain[position++];


				feedForward_mlp(W, layers, layer_count, inputCountPerSample, X);
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
					y[subimg] = YTrain[posY++];


				get_last_delta(X, layers, layer_count, y, delta);

				update_delta(W, X, layers, layer_count, delta, inputCountPerSample);

				Xout[img] = X[layer_count][layers[layer_count - 1]];

				update_W(W, X, layers, layer_count, inputCountPerSample, delta, alpha);
			}
			double* YT = new double[sampleCount];
			for (int k = 0; k < sampleCount; k++)
				YT[k] = (double)YTrain[k];

			double loss = mse_loss_mlp(YT, Xout, sampleCount);
			if (e % 10 == 0 || e == epochs - 1)
				printf("Epoch: %d loss: %f\n", e, loss);
		}

	}

	SUPEREXPORT void fit_mlp_regression(double*** W,
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
			if (l == 0)
				X[l] = new double[inputCountPerSample + 1];
			else
				X[l] = new double[layers[l - 1] + 1];
			X[l][0] = 1;
		}

		for (int e = 0; e < epochs; e++)
		{
			int position = 0;
			int posY = 0;
			double* Xout = new double[sampleCount];
			for (int img = 0; img < sampleCount; img++)
			{
				// Cahrger Xtrain => X[0]

				for (int n = 1; n < (inputCountPerSample + 1); n++)
					X[0][n] = Xtrain[position++];


				feedForward_mlp_regression(W, layers, layer_count, inputCountPerSample, X);
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
					y[subimg] = YTrain[posY++];


				get_last_delta_regression(X, layers, layer_count, y, delta);

				update_delta(W, X, layers, layer_count, delta, inputCountPerSample);

				Xout[img] = X[layer_count][layers[layer_count - 1]];

				update_W(W, X, layers, layer_count, inputCountPerSample, delta, alpha);
			}
			double* YT = new double[sampleCount];
			for (int k = 0; k < sampleCount; k++)
				YT[k] = (double)YTrain[k];

			double loss = mse_loss_mlp(YT, Xout, sampleCount);
			if (e % 10 == 0 || e == epochs - 1)
				printf("Epoch: %d loss: %f\n", e, loss);
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
				}
			}
		}
		return W;
	}
}