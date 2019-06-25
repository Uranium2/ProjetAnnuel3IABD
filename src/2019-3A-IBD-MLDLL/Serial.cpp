#include "Serial.h"

extern "C" {
	
	SUPEREXPORT void saveLinearModel(double* W, int inputCountPerSample, char* fileName)
	{
		std::ofstream myfile;
		myfile.open(fileName);

		myfile << inputCountPerSample << "\n";

		for (int i = 0; i < inputCountPerSample + 1; i++)
			myfile << W[i] << ",";

		myfile.close();
	}

	SUPEREXPORT int getInputCountPerSample(char* fileName)
	{
		std::ifstream  myfile;
		myfile.open(fileName);
		if (!myfile) {
			std::cout << "Unable to open file: " << fileName;
			exit(1);
		}
		std::string line = "";
		std::getline(myfile, line);
		std::string content = "";
		std::istringstream lineStream(line);
		
		std::getline(lineStream, content, ',');
		std::istringstream number(content);
		int inputCountPerSample = 0;
		number >> inputCountPerSample;

		myfile.close();
		return inputCountPerSample;
	}
	SUPEREXPORT double* loadLinearModel(char* fileName)
	{
		std::ifstream  myfile;
		myfile.open(fileName);
		if (!myfile) {
			std::cout << "Unable to open file: " << fileName;
			exit(1);
		}
		std::string line = "";
		std::getline(myfile, line);
		std::string content = "";
		std::istringstream lineStream(line);
		
		std::getline(lineStream, content, ',');
		std::istringstream number(content);
		int inputCountPerSample = 0;
		number >> inputCountPerSample;

		double* W = new double[inputCountPerSample + 1];
		for (int i = 0; i < inputCountPerSample + 1; i++)
		{
			myfile >> W[i];
			myfile.get();
		}

		myfile.close();
		return W;
	}
	
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
		
	}

	SUPEREXPORT int getLayer_count(char* fileName)
	{
		int layer_count = 0;
		std::ifstream  myfile;
		myfile.open(fileName);
		if (!myfile) {
			std::cout << "Unable to open file: " << fileName;
			exit(1);
		}
		std::string line = "";
		std::getline(myfile, line);
		std::string content = "";
		std::istringstream lineStream(line);
		std::istringstream lineStreamCopy(line);
		while (std::getline(lineStream, content, ',')) // Get nb layers
			layer_count++;

		myfile.close();	
		return layer_count;
	}
	SUPEREXPORT int* getLayers(char* fileName)
	{

		int layer_count = 0;
		std::ifstream  myfile;
		myfile.open(fileName);
		if (!myfile) {
			std::cout << "Unable to open file: " << fileName;
			exit(1);
		}
		std::string line = "";
		std::getline(myfile, line);
		std::string content = "";
		std::istringstream lineStream(line);
		std::istringstream lineStreamCopy(line);
		while (std::getline(lineStream, content, ',')) // Get nb layers
			layer_count++;

		int* layers = new int[layer_count];
		int i = 0;
		while (std::getline(lineStreamCopy, content, ',')) // Insert to array
		{
			std::istringstream number(content);
			number >> layers[i++];
		}
		myfile.close();
		return layers;
	}
	SUPEREXPORT double*** loadModel(char* fileName)
	{
		int layer_count = 0;
		std::ifstream  myfile;
		myfile.open(fileName);
		if (!myfile) {
			std::cout << "Unable to open file: " << fileName;
			exit(1);
		}
		std::string line = "";
		std::getline(myfile, line);
		std::string content = "";
		std::istringstream lineStream(line);
		std::istringstream lineStreamCopy(line);
		while (std::getline(lineStream, content, ',')) // Get nb layers
			layer_count++;


		int* layers = new int[layer_count];
		int i = 0;
		while (std::getline(lineStreamCopy, content, ',')) // Insert to array
		{
			std::istringstream number(content);
			number >> layers[i++];
		}

		double*** W = new double** [layer_count];

		for (int l = 1; l < layer_count; l++)
		{
			W[l] = new double* [layers[l] + 1];
			for (int j = 1; j < layers[l] + 1; j++)
			{
				W[l][j] = new double[layers[l - 1] + 1];
				for (int i = 0; i < (layers[l - 1] + 1); i++)
				{
					myfile >> W[l][j][i];
					myfile.get();
				}
			}
		}
		myfile.close();
		return W;
	}
}