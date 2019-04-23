#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <random>


typedef enum {
	LINEAR,
	SIGN,
	SIGMOID,
	TANH
} ACTIVATION;

typedef struct neuron
{
	double* inputs;
	double* weights;
	double output;
	int nbInputs;
	ACTIVATION typeActivation;
	double bias;
} Neuron;

Neuron* createNeuron(double* inputs, ACTIVATION typeActivation, double bias, int Nbinputs);

void feedForward(Neuron* neuron);

double deriv_sigmoid(double x);