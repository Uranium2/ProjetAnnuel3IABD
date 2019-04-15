#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "neuron.h"

#define KNRM "\x1B[0m"
#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"
#define KYEL "\x1B[33m"
#define KBLU "\x1B[34m"
#define KMAG "\x1B[35m"
#define KCYN "\x1B[36m"
#define KWHT "\x1B[37m"

typedef struct layer
{
	Neuron** neurons;
	int nbNeurons;
} Layer;

typedef struct neuralNet
{
	double* inputs;
	Layer** Layers; // Matrix of neurons
	int nbLayers;
	int* sizeLayers;
} NeuralNet;

NeuralNet* buildNeuralNet(double* inputs, int nbLayers, int* sizeLayers, int nbInputs);

double mse_loss(double* v_true, double* v_given, int nb_elem);

void printNN(NeuralNet* nn);

void feedForwadAll(NeuralNet* nn);