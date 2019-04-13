#include "Neuron.h"
#include <functional>
#include <chrono>

double* setRandomWeights(int lower, int upper, int nbInputs)
{
	auto W = new double[nbInputs + 1];
	double low = -1.0;
	double up = 1.0;
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);

	std::uniform_real_distribution<double> distribution(low, up);
	for (int i = 0; i < nbInputs + 1; i++)
	{
		W[i] = distribution(generator);
	}
	// TODO : initialisation random [-1,1]
	return W;
}

Neuron* createNeuron(double* inputs, int typeActivation, double bias, int Nbinputs)
{
	Neuron* n = (Neuron*)malloc(sizeof(Neuron));
	n->inputs = inputs;
	n->nbInputs = Nbinputs;
	n->weights = (double*)malloc(sizeof(float) * Nbinputs);
	n->typeActivation = typeActivation;
	n->bias = bias;
	n->weights = new double[Nbinputs + 1];
	return n;
}

double* getWeightedInput(double* inputs, double* weights, int nbInputs)
{
	double* mult = (double*)malloc(sizeof(double) * nbInputs);
	for (int i = 0; i < nbInputs; i++)
	{
		mult[i] = inputs[i] * weights[i];
	}
	return mult;
}

double sumWeightedInput(double* weightedInput, int nbInputs, int bias)
{
	double sum = 0;

	for (int i = 0; i < nbInputs; i++)
	{
		sum += weightedInput[i];
	}
	return sum + bias;
}

double hyperTan(double x)
{
	return (double)((1 - exp(-2 * x)) / (1 + exp(-2 * x)));
}

float sigmoid(float x)
{
	return (double)(1 / (1 + exp(-x)));
}

double deriv_sigmoid(double x)
{
	double sig = sigmoid(x);
	return sig * (1 - sig);
}

double linear(double x)
{
	return (double)x;
}

double reLu(double x)
{
	if (x < 0)
		return 0;
	return linear(x);
}

float leackyReLu(float x)
{
	if (x < 0)
		return linear(x)* 0.01;
	return linear(x);
}

double activateFunction(double x, int type)
{
	switch (type)
	{
	case 0:
		return linear(x);
	case 1:
		return sigmoid(x);
	case 2:
		return hyperTan(x);
	case 3:
		return reLu(x);
	case 4:
		return leackyReLu(x);
	default:
		return sigmoid(x);
	}
}

void feedForward(Neuron * neuron)
{
	neuron->output = activateFunction(
		sumWeightedInput(
			getWeightedInput(
				neuron->inputs, neuron->weights, neuron->nbInputs),
			neuron->nbInputs, neuron->bias),
		neuron->typeActivation);
}