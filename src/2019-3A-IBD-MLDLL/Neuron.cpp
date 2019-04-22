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

Neuron* createNeuron(double* inputs, int typeActivation, double bias, int Nbinputs)
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
		lol
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
	auto weightedInput = getWeightedInput(neuron->inputs, neuron->weights, neuron->nbInputs);
	neuron->output = activateFunction(
		sumWeightedInput(
			weightedInput,
			neuron->nbInputs, neuron->bias),
		neuron->typeActivation);
}