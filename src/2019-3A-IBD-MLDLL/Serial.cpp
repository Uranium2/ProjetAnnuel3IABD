#include "Serial.h"

extern "C" {
	SUPEREXPORT void saveModel(double*** W, int* layers, int layer_count, char* fileName)
	{
		std::ofstream myfile;
		myfile.open(fileName);

		for (int i = 0; i < layer_count; i++)
		{
			myfile << layers[i] << ",";
		}
		myfile << "\n";
		for (int l = 1; l < layer_count; l++)
		{
			for (int j = 1; j < layers[l] + 1; j++)
			{
				for (int i = 0; i < (layers[l - 1] + 1); i++)
					myfile << W[l][j][i] << ",";
			}
		}
		myfile.close();
	}
}