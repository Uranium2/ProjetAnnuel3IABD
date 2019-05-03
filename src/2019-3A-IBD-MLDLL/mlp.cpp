#include <iostream>
#include <chrono>
#include <random>
#include <limits>
#include "Mlp.h"
#include <cmath>

void print_array(double** arr, int sampleCount, int inputCountPerSample, bool hasBiais)
{
	if (hasBiais)
		inputCountPerSample++;

	int dimX = sampleCount;
	for (int x = 0; x < dimX; ++x) {
		int dimY = inputCountPerSample;
		for (int y = 0; y < dimY; ++y) {
			arr[x][y] = -1;
			std::cout << arr[x][y] << " ";
		}
		std::cout << "\n";
	}
}
double squared_error(double v_true, double v_given)
{
	//std::cout << "expected: " << v_true << " have: " << v_given << "\n";
	return pow(v_true - v_given, 2);
}
double mse_loss(double* v_true, double* v_given, int nb_elem, std::vector<int> myImageIndex)
{
	//sstd::cout << "calc mse loss\n";
	double res = 0.0;
	for (int i = 0; i < nb_elem; i++)
	{
		res += squared_error(v_true[myImageIndex[i]], v_given[myImageIndex[i]]);
	}
	return res / nb_elem;
}
double** init_Xall(int* layers, int layer_count)
{
	int dimX = layer_count + 1;
	double** Xall = new double* [dimX];

	for (int x = 0; x < dimX; ++x) {
		int dimY = layers[x] + 1;
		Xall[x] = new double[dimY];
	}
	return Xall;
}
void print_Xall(double** Xall, int* layers, int layer_count)
{
	int dimX = layer_count;

	for (int x = 0; x < dimX; ++x) {
		int dimY = layers[x] + 1;
		for (int y = 0; y < dimY; y++)
		{
			std::cout << "Xall[" << x << "][" << y << "] = " << Xall[x][y] << " ";
		}
		std::cout << "\n";
	}
}
double sigmoid(double x)
{
	return std::tanh(x);
}
double weighted_sum(double w, double x)
{
	return w * x;
}
void feedForwardMLP(MLP* mlp, int inputCountPerSample)
{
	int dimY = 0;
	int dimZ = 0;
	for (int l = 1; l < mlp->layer_count + 1; l++)
	{
		if (l == 1)
			dimY = inputCountPerSample;
		else
			dimY = mlp->layers[l - 1];
		for (int j = 0; j < dimY + 1; j++)
		{
			if (l == 1)
				dimZ = inputCountPerSample;
			else
				dimZ = mlp->layers[l - 2];
			double res = 0.0;
			for (int i = 1; i < dimZ + 1; i++)
			{
				res += weighted_sum(mlp->W[l][i][j], mlp->Xall[l - 1][j]);
			}
			mlp->Xall[l][j] = sigmoid(res);
		}
	}
}
void printMLP(double*** W, int* layers, int layer_count, int inputCountPerSample)
{
	int dimY = 0;
	int dimZ = 0;
	for (int l = 1; l < layer_count + 1; l++)
	{
		if (l == 1)
			dimY = inputCountPerSample;
		else
			dimY = layers[l - 2];
		for (int i = 0; i < dimY + 1; i++)
		{
			if (l == 1)
				dimZ = inputCountPerSample;
			else
				dimZ = layers[l - 1];
			for (int j = 1; j < dimZ + 1; j++)
			{
				std::cout << "mlp[" << l << "]" <<
					"[" << i << "]" <<
					"[" << j << "]: " << W[l][i][j] << " ";
			}
			std::cout << "\n";
		}
		std::cout << "\n";
	}
}
double*** create_mlp_model(int* layers, int layer_count, int inputCountPerSample)
{
	double low = -1.0;
	double up = 1.0;
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::uniform_real_distribution<double> distribution(low, up);

	double*** mlp = new double** [layer_count
		+ 1];
	int dimY = 0;
	int dimZ = 0;
	for (int l = 1; l < layer_count + 1; l++)
	{
		if (l == 1)
			dimY = inputCountPerSample;
		else
			dimY = layers[l - 2];
		mlp[l] = new double* [dimY + 1];
		for (int i = 0; i < dimY + 1; i++)
		{
			if (l == 1)
				dimZ = inputCountPerSample;
			else
				dimZ = layers[l - 1];
			mlp[l][i] = new double[dimZ + 1];
			for (int j = 1; j < dimZ + 1; j++)
			{
				mlp[l][i][j] = distribution(generator);
			}
		}
	}

	return mlp;
}
double getPrediction(MLP * mlp, int inputCountPerSample) {
	feedForwardMLP(mlp, inputCountPerSample);
	return mlp->Xall[1][1];
}
double** init_delta(int* layers, int layer_count)
{
	int dimX = layer_count + 1;
	double** delta;
	delta = new double* [dimX];

	for (int l = 1; l < dimX; l++) {
		int dimY = layers[l - 1] + 1;
		delta[l] = new double[dimY];
		for (int y = 0; y < dimY; ++y) {
			delta[l][y] = 0;
			std::cout << "delta[" << l << "][" << y << "] = " << delta[l][y] << " ";
		}
		std::cout << "\n";
	}
	return delta;
}
double** list_to_array(double* list, int sampleCount, int inputCountPerSample, bool addBiais)
{
	int dimX = sampleCount;
	double** arr;
	arr = new double* [dimX];
	int pos = 0;
	if (addBiais)
		inputCountPerSample++;

	for (int x = 0; x < dimX; ++x) {
		int dimY = inputCountPerSample;
		arr[x] = new double[dimY];
		for (int y = 0; y < dimY; ++y) {
			if (y == 0 && addBiais)
				arr[x][y] = 1;
			else
			{
				arr[x][y] = list[pos];
				pos++;
			}
		}
	}
	return arr;
}
MLP* build_mlp(double*** W, int* layers, int layer_count, double* XTrain, double* YTrain, int sampleCount, int inputCountPerSample, double alpha, int epochs)
{
	MLP* mlp = (MLP*)malloc(sizeof(MLP));
	if (mlp == NULL)
		return NULL;
	else
	{
		mlp->W = W;
		mlp->layers = layers;
		mlp->layer_count = layer_count;
		mlp->XTrain = XTrain;
		mlp->YTrain = YTrain;
		mlp->delta = init_delta(layers, layer_count);
		mlp->Xall = init_Xall(layers, layer_count);
		mlp->Yall = list_to_array(YTrain, sampleCount, 1, false);
		mlp->alpha = alpha;
		mlp->epochs = epochs;
		for (int l = 0; l < mlp->layer_count; l++) // bias
		{
			mlp->Xall[l][0] = 1;
		}
	}
	return mlp;
}
void delta_cal(MLP * mlp, int sampleCount, int inputCountPerSample)
{
	int dimY = 0;
	int dimZ = 0;
	int last_layer = mlp->layer_count - 1;
	std::vector<int> myImageIndex;
	auto rng = std::default_random_engine{};

	for (int i = 0; i < sampleCount; i++) // Create ordered vector
		myImageIndex.push_back(i);

	std::shuffle(std::begin(myImageIndex), std::end(myImageIndex), rng); //shuffle indexes images

	for (int e = 0; e < mlp->epochs; e++)
	{
		double* Xout = new double[(double)sampleCount];
		for (int img = 0; img < sampleCount; img++)
		{
			//std::cout << "new image\n";
			for (int n = 1; n < inputCountPerSample + 1; n++) // Shuffle index
				mlp->Xall[0][n] = mlp->XTrain[(inputCountPerSample * myImageIndex[img]) + n - 1];

			feedForwardMLP(mlp, inputCountPerSample); // Update outputs
			for (int j = 0; j < mlp->layers[last_layer] + 1; j++) // Update Delta of last layer
			{
				//std::cout << "mlp->delta[" << mlp->layers[last_layer] + 1 << " ][" << j << "]\n";
				mlp->delta[mlp->layers[last_layer] + 1][j] =
					(1 - std::pow(mlp->Xall[last_layer][j], 2) *
					(mlp->Xall[last_layer][j] - mlp->Yall[myImageIndex[img]][j]));
			}

			for (int l = mlp->layer_count; l > 1; l--)
			{

				for (int i = 0; i < mlp->layers[l - 2] + 1; i++)
				{
					double res = 0.0;
					//std::cout << "delta[" << l - 1 << "][" << i << "] = ";
					for (int j = 1; j < mlp->layers[l - 1] + 1; j++)
					{
						//std::cout << "W[" << l << "]" <<
							//"[" << i << "]" <<
							//"[" << j << "]: " << " * ";
						//std::cout << "delta[" << l << "][" << j << "]\n";
						res += weighted_sum(mlp->W[l][i][j], mlp->delta[l][j]);

					}
					mlp->delta[l - 1][i] = 1 - std::pow(mlp->Xall[l - 1][i], 2) * res;
				}
			}

			for (int l = 1; l < mlp->layer_count + 1; l++)
			{
				if (l == 1)
					dimY = inputCountPerSample;
				else
					dimY = mlp->layers[l - 2];
				for (int i = 0; i < dimY + 1; i++)
				{
					if (l == 1)
						dimZ = inputCountPerSample;
					else
						dimZ = mlp->layers[l - 1];
					for (int j = 1; j < dimZ + 1; j++)
					{
						mlp->W[l][i][j] = mlp->W[l][i][j] - mlp->alpha * (mlp->Xall[l - 1][i] * mlp->delta[l][j]);
					}
				}
			}
			Xout[img] = getPrediction(mlp, inputCountPerSample);
		}


		if (e % 1 == 0 || e == mlp->epochs - 1)
		{
			double loss = mse_loss(mlp->YTrain, Xout, sampleCount, myImageIndex);
			printMLP(mlp->W, mlp->layers, mlp->layer_count, inputCountPerSample);

			printf("Epoch: %d loss: %f\n", e, loss);
			std::cout << "----------------------------------------------------------------------------------\n";
		}
	}
	feedForwardMLP(mlp, inputCountPerSample);
}
void print_inputs(double** inputs, int* layers, int layer_count)
{
	for (int x = 0; x < layer_count; ++x) {
		int dimY = layers[x] + 1;
		for (int y = 0; y < dimY; ++y) {
			std::cout << inputs[x][y] << " ";
		}
		std::cout << "\n";
	}
}
double*** fit_mlp_classification(double*** W,
	double* XTrain,
	double* YTrain,
	int* layers,
	int layer_count,
	int sampleCount,
	int inputCountPerSample,
	double alpha,
	int epochs)
{
	MLP* mlp = build_mlp(W, layers, layer_count, XTrain, YTrain, sampleCount, inputCountPerSample, alpha, epochs);
	delta_cal(mlp, sampleCount, inputCountPerSample);
	return W;
}

int main2() {
	int layers[2] = { 2, 1 };
	int layer_count = 2;
	int inputCountPerSample = 2;
	double*** W = create_mlp_model(layers, layer_count, inputCountPerSample);
	double alpha = 0.5;
	int epochs = 10;
	printMLP(W, layers, layer_count, inputCountPerSample);
	double Xtrain[8] = { 0.0, 0.0,
						0.0, 1.0,
						1.0, 0.0,
						1.0, 1.0 };
	double YTrain[4] = { 1, -1, -1, 1 };
	fit_mlp_classification(W, Xtrain, YTrain, layers, layer_count, 4, inputCountPerSample, alpha, epochs);
	return 0;
}
