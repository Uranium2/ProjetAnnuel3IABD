#include "NeuralNet.h"

Layer* initLayerNeuron(int nbNeurons)
{
	Layer* l = (Layer*)malloc(sizeof(Layer));
	l->neurons = (Neuron**)malloc(sizeof(Neuron) * nbNeurons);
	l->nbNeurons = nbNeurons;
	return l;
}

NeuralNet* initNeuralNet(double* inputs, int nbLayers, int* sizeLayers)
{
	NeuralNet* nn = (NeuralNet*)malloc(sizeof(NeuralNet));
	nn->inputs = inputs;
	nn->nbLayers = nbLayers;
	nn->sizeLayers = sizeLayers;
	nn->Layers = (Layer**)malloc(sizeof(Layer*) * nbLayers);
	return nn;
}

NeuralNet* buildNeuralNet(double* inputs, int nbLayers, int* sizeLayers)
{
	NeuralNet* nn = initNeuralNet(inputs, nbLayers, sizeLayers);
	for (int i = 0; i < nbLayers; i++)
	{
		Layer* l = initLayerNeuron(sizeLayers[i]);
		for (int j = 0; j < sizeLayers[i]; j++)
		{
			if (i == 0)
			{
				l->neurons[j] = createNeuron(inputs, 2, 0, sizeLayers[i]);
				feedForward(l->neurons[j]);
				continue;
			}
			double* in = (double*)malloc(sizeof(float) * sizeLayers[i - 1]);
			for (int k = 0; k < sizeLayers[i]; k++)
			{
				for (int l = 0; l < sizeLayers[i - 1]; l++)
				{
					in[k] = nn->Layers[i - 1]->neurons[l]->output;
				}
			}

			l->neurons[j] = createNeuron(in, 2, 0, sizeLayers[i]);
			feedForward(l->neurons[j]);
		}
		nn->Layers[i] = l;
	}
	return nn;
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

void printNN(NeuralNet* nn)
{
	printf("Input Layer\n");
	for (int i = 0; i < nn->nbLayers; i++)
	{
		for (int j = 0; j < nn->sizeLayers[i]; j++)
		{
			system("Color 0");
			std::cout << "x ";
			for (int k = 0; k < nn->Layers[i]->nbNeurons; k++)
			{
				std::cout << KRED << nn->Layers[i]->neurons[j]->inputs[k] << " ";
				std::cout << KBLU << nn->Layers[i]->neurons[j]->weights[k] << " ";
				std::cout << KGRN << nn->Layers[i]->neurons[j]->bias << " ";
			}
		}
		printf("\n");
	}
	std::cout << KWHT;
	printf("Output Layer\n");
}

void feedForwadAll(NeuralNet* nn)
{
	for (int i = 0; i < nn->nbLayers; i++)
	{
		for (int j = 0; j < nn->sizeLayers[i]; j++)
		{
			for (int k = 0; k < nn->Layers[i]->nbNeurons; k++)
			{
				feedForward(nn->Layers[i]->neurons[k]);
			}
		}
	}
}