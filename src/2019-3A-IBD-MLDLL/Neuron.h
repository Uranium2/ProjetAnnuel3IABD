#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <random>


typedef struct neuron
{
	double* inputs;
	double* weights;
	double output;
	int nbInputs;
	int typeActivation;
	double bias;
} Neuron;

Neuron* createNeuron(double* inputs, int typeActivation, double bias, int Nbinputs);

void feedForward(Neuron* neuron);

double deriv_sigmoid(double x);