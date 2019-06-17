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
			res += squared_error_mlp(v_true[i], v_given[i]);

		return res / nb_elem;
	}

	SUPEREXPORT double* predict_mlp_classification(double*** W, int* layers, int layer_count, int inputCountPerSample, double* Xinput)
	{

		double** X = new double* [layer_count];

		for (int l = 0; l < layer_count; l++)
		{
			if (l == 0)
			{
				X[l] = new double[inputCountPerSample + 1];
				int pos = 0;
				for (int input = 1; input < inputCountPerSample + 1; input++)
					X[l][input] = Xinput[pos++];
			}
			else
				X[l] = new double[layers[l] + 1];
			X[l][0] = 1;
		}


		for (int l = 1; l < (layer_count); l++)
		{
			for (int j = 1; j < (layers[l] + 1); j++)
			{
				int y = 0;
				if (l == 1)
					y = inputCountPerSample + 1;
				else
					y = layers[l - 1] + 1;
				double res = 0.0;
				for (int i = 0; i < y; i++)
				{
					//std::cout << "X[" << l - 1 << "][" << i << "] " << X[l - 1][i] << "\n";
					res += W[l][i][j] * X[l - 1][i];
				}
				X[l][j] = std::tanh(res);
			}
		}
		return X[layer_count - 1];
	}

	SUPEREXPORT double* predict_mlp_regression(double*** W, int* layers, int layer_count, int inputCountPerSample, double* Xinput)
	{
		double** X = new double* [layer_count];

		for (int l = 0; l < layer_count; l++)
		{
			if (l == 0)
				X[l] = new double[inputCountPerSample + 1];
			else
				X[l] = new double[layers[l] + 1];
			X[l][0] = 1;
		}

		for (int l = 1; l < (layer_count); l++)
		{
			for (int j = 1; j < (layers[l] + 1); j++)
			{
				int y = 0;
				if (l == 1)
					y = inputCountPerSample + 1;
				else
					y = layers[l - 1] + 1;
				double res = 0.0;
				for (int i = 0; i < y; i++)
					res += W[l][i][j] * X[l - 1][i];
				if (l == layer_count)
					X[l][j] = res;
				else
					X[l][j] = std::tanh(res);
			}
		}
		return X[layer_count - 1];
	}

	void feedForward_mlp(double*** W, int* layers, int layer_count, int inputCountPerSample, double** X)
	{
		for (int l = 1; l < (layer_count); l++)
		{
			for (int j = 1; j < (layers[l] + 1); j++)
			{
				int y = 0;
				if (l == 1)
					y = inputCountPerSample + 1;
				else
					y = layers[l - 1] + 1;
				double res = 0.0;
				for (int i = 0; i < y; i++)
				{
					res += W[l][i][j] * X[l - 1][i];
					//std::cout << "feedforward = " << "W[" << l << "][" << i << "][" << j << "] " << W[l][i][j] << " * " <<
					//	"X[" << l - 1 << "][" << i << "] " << X[l - 1][i] << "\n";
				}
				
				X[l][j] = std::tanh(res);
				//std::cout << "X[" << l << "][" << j << "] " << X[l][j]  <<"\n";
			}
		}
	}

	void feedForward_mlp_regression(double*** W, int* layers, int layer_count, int inputCountPerSample, double** X)
	{
		for (int l = 1; l < (layer_count); l++)
		{
			for (int j = 1; j < (layers[l] + 1); j++)
			{
				int y = 0;
				if (l == 1)
					y = inputCountPerSample + 1;
				else
					y = layers[l - 1] + 1;
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

	void update_delta(double*** W, double** X, int* layers, int layer_count, double** delta, int inputCountPerSample)
	{
		for (int l = layer_count - 1; l > 0; l--)
		{
			//std::cout << "l = " << l << "\n";
			int y = 0;
			if (l == 1)
				y = inputCountPerSample + 1;
			else
				y = layers[l - 1] + 1;
			//std::cout << "y = " << y << "\n";
			for (int i = 1; i < y; i++)
			{
				double res = 0.0;
				for (int j = 1; j < (layers[l] + 1); j++)
				{
					res += W[l][i][j] * delta[l][j];
					//std::cout << "delta[l] in update delta l = " << l << "delta " << delta[l][i] << "\n";
					//std::cout << "updateDelta = " << W[l][i][j] << " * " << delta[l][i] << "\n";
				}
				//std::cout << "\n";

				delta[l - 1][i] = (1 - std::pow(X[l - 1][i], 2)) * res;
				//std::cout << "delta[" << l - 1 << "][" << i << "]" << delta[l -1][i] << "\n";

			}
		}
	}

	void update_W(double*** W, double** X, int* layers, int layer_count, int inputCountPerSample, double** delta, double alpha) {
		for (int l = 1; l < layer_count; l++) {

			//std::cout << "l = " << l << "\n";
			for (int j = 1; j < layers[l] + 1; j++)
			{
				int y = 0;
				if (l == 1)
					y = inputCountPerSample + 1;
				else
					y = layers[l - 1] + 1;
				//std::cout << "\t i = " << i << "\n";
				for (int i = 0; i < y; i++)
				{
					//std::cout << "\t\t j = " << j << "\n";
					W[l][i][j] = W[l][i][j] - (alpha * X[l - 1][i] * delta[l][j]);
					//std::cout << "UpdateW " << W[l][i][j] << " - " << delta[l][j] << "\n";
				}
			}
		}
	}

	void get_last_delta(double** X, int* layers, int layer_count, int* Y, double** delta) {
		int L = layer_count - 1;
		for (int j = 1; j < layers[L] + 1; j++)
		{
			delta[L][j] = (1 - std::pow(X[L][j], 2)) * (X[L][j] - Y[j - 1]);
			//std::cout << "delta[" << L << "][" << j << "]"  << delta[L][j] << "\n";
		}
	}

	void get_last_delta_regression(double** X, int* layers, int layer_count, int* Y, double** delta) {
		int L = layer_count - 1;
		for (int j = 1; j < layers[L] + 1; j++)
			delta[L][j] = (X[L][j] - Y[j - 1]);
	}

	SUPEREXPORT void fit_mlp_classification(double*** W,
		double* Xtrain,
		int* YTrain,
		int* layers,
		int layer_count,
		int sampleCount,
		int inputCountPerSample,
		double alpha,
		int epochs)
	{
		double** X = new double* [layer_count];
		double** delta = new double* [layer_count];

		for (int l = 0; l < layer_count; l++)
		{
			if (l == 0)
			{
				X[l] = new double[inputCountPerSample + 1];
				delta[l] = new double[inputCountPerSample + 1];
			}
			else
			{
				X[l] = new double[layers[l] + 1];
				delta[l] = new double[layers[l] + 1];
			}
			X[l][0] = 1;
		}

		std::vector<int> myImageIndex;
		auto rng = std::default_random_engine{};

		for (int i = 0; i < sampleCount; i++) // Create ordered vector
			myImageIndex.push_back(i);

		

		int* y = new int[layers[layer_count - 1]];

		double* YT = new double[sampleCount];

		double* Xout0 = new double[sampleCount];
		double* Xout1 = new double[sampleCount];
		double* Xout2 = new double[sampleCount];

		for (int e = 0; e < epochs; e++)
		{

			std::shuffle(std::begin(myImageIndex), std::end(myImageIndex), rng); //shuffle indexes images at each epoch

			for (int img = 0; img < sampleCount; img++)
			{
				// Load Inputs
				for (int n = 1; n < (inputCountPerSample + 1); n++)
				{
					X[0][n] = Xtrain[(inputCountPerSample * myImageIndex[img]) + n - 1];
					//std::cout << "X[0][" << 1 << "]" << X[0][n] << "\n";
				}
				//std::cout << "\n";



				feedForward_mlp(W, layers, layer_count, inputCountPerSample, X);

				for (int subimg = 0; subimg < layers[layer_count - 1]; subimg++) // Load Random Image
				{
					y[subimg] = YTrain[(layers[layer_count - 1] * myImageIndex[img]) + subimg];
					//std::cout << y[subimg] << "\n";
				}

				get_last_delta(X, layers, layer_count, y, delta);

				update_delta(W, X, layers, layer_count, delta, inputCountPerSample);

				Xout0[myImageIndex[img]] = X[layer_count - 1][1];
				Xout1[myImageIndex[img]] = X[layer_count - 1][2];
				Xout2[myImageIndex[img]] = X[layer_count - 1][3];

				update_W(W, X, layers, layer_count, inputCountPerSample, delta, alpha);
			}

			for (int k = 0; k < sampleCount; k++)
				YT[k] = (double)YTrain[k];

			double loss0 = mse_loss_mlp(YT, Xout0, sampleCount);
			if (e % 1000 == 0 || e == epochs - 1)
				printf("Epoch: %d loss: %f\n", e, loss0);
			double loss1 = mse_loss_mlp(YT, Xout1, sampleCount);
			if (e % 1000 == 0 || e == epochs - 1)
				printf("Epoch: %d loss: %f\n", e, loss1);
			double loss2 = mse_loss_mlp(YT, Xout2, sampleCount);
			if (e % 1000 == 0 || e == epochs - 1)
				printf("Epoch: %d loss: %f\n", e, loss2);
		}
		//delete[] X, myImageIndex, Xout, YT, delta;
	}

	SUPEREXPORT void fit_mlp_regression(double*** W,
		double* Xtrain,
		int* YTrain,
		int* layers,
		int layer_count,
		int sampleCount,
		int inputCountPerSample,
		double alpha,
		int epochs)
	{
		double** X = new double* [layer_count];
		double** delta = new double* [layer_count];

		for (int l = 0; l < layer_count; l++)
		{
			if (l == 0)
			{
				X[l] = new double[inputCountPerSample + 1];
				delta[l] = new double[inputCountPerSample + 1];
			}
			else
			{
				X[l] = new double[layers[l] + 1];
				delta[l] = new double[layers[l] + 1];
			}
			X[l][0] = 1;
		}

		std::vector<int> myImageIndex;
		auto rng = std::default_random_engine{};

		for (int i = 0; i < sampleCount; i++) // Create ordered vector
			myImageIndex.push_back(i);

		std::shuffle(std::begin(myImageIndex), std::end(myImageIndex), rng); //shuffle indexes images

		int* y = new int[layers[layer_count - 1]];

		double* YT = new double[sampleCount];

		double* Xout = new double[sampleCount];

		for (int e = 0; e < epochs; e++)
		{

			for (int img = 0; img < sampleCount; img++)
			{
				// Load Inputs
				for (int n = 1; n < (inputCountPerSample + 1); n++)
					X[0][n] = Xtrain[(inputCountPerSample * myImageIndex[img]) + n - 1];


				feedForward_mlp_regression(W, layers, layer_count, inputCountPerSample, X);

				for (int subimg = 0; subimg < layers[layer_count - 1]; subimg++) // Load Random Image
					y[subimg] = YTrain[(layers[layer_count - 1] * myImageIndex[img]) + subimg];

				get_last_delta_regression(X, layers, layer_count, y, delta);

				update_delta(W, X, layers, layer_count, delta, inputCountPerSample);

				Xout[myImageIndex[img]] = X[layer_count - 1][1];

				update_W(W, X, layers, layer_count, inputCountPerSample, delta, alpha);
			}

			for (int k = 0; k < sampleCount; k++)
				YT[k] = (double)YTrain[k];

			double loss = mse_loss_mlp(YT, Xout, sampleCount);
			if (e % 1000 == 0 || e == epochs - 1)
				printf("Epoch: %d loss: %f\n", e, loss);
		}
		//delete[] X, myImageIndex, Xout, YT, delta;
	}


	SUPEREXPORT double*** create_mlp_model(int* layers, int layer_count, int inputCountPerSample)
	{
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::default_random_engine generator(seed);
		std::uniform_real_distribution<double> distribution(-1, 1);

		double*** W = new double** [layer_count];

		int k = 0;
		for (int l = 1; l < (layer_count); l++)
		{
			int y = 0;
			if (l == 1)
				y = inputCountPerSample + 1;
			else
				y = layers[l - 1] + 1;
			W[l] = new double* [y];
			//std::cout << "l = " << l << "\n";
			for (int i = 0; i < y; i++)
			{
				//std::cout << "\ti = " << i << "\n";
				W[l][i] = new double[layers[l] + 1];
				for (int j = 1; j < (layers[l] + 1); j++)
				{
					//std::cout << "\t\tj = " << j << "\n";
					W[l][i][j] = distribution(generator);
					//std::cout << "W[" << l << "][" << i << "][" << j << "] = " << W[l][i][j] << " ";
				}
				//std::cout << "\n";
			}
			//std::cout << "\n";
		}

		return W;
	}
}