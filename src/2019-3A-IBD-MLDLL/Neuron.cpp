#include "Neuron.h"
#include <functional>
#include <chrono>

double* setRandomWeights(int lower, int upper, int nbInputs)
{
	auto W = new double[(double)nbInputs];
	double low = -1.0;
	double up = 1.0;
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);

	std::uniform_real_distribution<double> distribution(low, up);
	for (int i = 0; i < nbInputs; i++)
	{
		W[i] = distribution(generator);
	}
	// TODO : initialisation random [-1,1]
	return W;
}

Neuron* createNeuron(double* inputs, ACTIVATION typeActivation, double bias, int Nbinputs)
{
	Neuron* n = (Neuron*)malloc(sizeof(Neuron));
	if (n == NULL)
		exit(-99);
	n->inputs = inputs;
	n->nbInputs = Nbinputs;
	n->weights = (double*)malloc(sizeof(double) * Nbinputs);
	n->typeActivation = typeActivation;
	n->bias = bias;
	n->weights = new double[(double)Nbinputs];
	return n;
}

double* getWeightedInput(double* inputs, double* weights, int nbInputs)
{
	double* mult = new double[(double)nbInputs + 1];
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

double sign(double x)
{
	if (x >= 0)
		return x;
	return 0;
}

double hyperTan(double x)
{
	return std::tanh(x);
}

double sigmoid(double x)
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

double leackyReLu(double x)
{
	if (x < 0)
		return linear(x)* 0.01;
	return linear(x);
}

double activateFunction(double x, ACTIVATION type)
{
	switch (type)
	{
	case LINEAR:
		return linear(x);
	case SIGN:
		return sign(x);
	case SIGMOID:
		return sigmoid(x);
	case TANH:
		return hyperTan(x);
	default:
		return sign(x);
	}
}

void feedForward(Neuron * neuron)
{
	auto weightedInput = getWeightedInput(neuron->inputs, neuron->weights, neuron->nbInputs);
	neuron->output = activateFunction(
		sumWeightedInput(
			weightedInput,
			neuron->nbInputs, neuron->bias),
		neuron->typeActivation);
}