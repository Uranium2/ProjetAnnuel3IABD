#include "Kmeans.h"


extern "C" {

	double get_distanceK(double* Xpredict, double* Xn, int inputCountPerSample) {

		// l2-norm
		double accum = 0.;

		for (int i = 0; i < inputCountPerSample; ++i) {
			double res = Xpredict[i] - Xn[i];
			accum += res * res;
		}

		double norm = sqrt(accum);
		return norm;
	}

	double* select_random_k(int K, double* Xtrain, int sampleCount, int inputCountPerSample) {
		double* Kmeans = new double[K * inputCountPerSample];
			
		std::vector<int> inputIndex;
		auto rng = std::default_random_engine{};

		for (int i = 0; i < sampleCount; i++) // Create ordered vector
			inputIndex.push_back(i);

		std::shuffle(std::begin(inputIndex), std::end(inputIndex), rng); //shuffle indexes inputs

		int pos = 0;
		for (int i = 0; i < K; i++)
		{
			for (int n = 0; n < inputCountPerSample; n++)
			{
				Kmeans[pos] = Xtrain[(inputCountPerSample * inputIndex[i]) + n - 1];
				std::cout << Kmeans[pos] << " ";
				pos++;
			}
			std::cout << "\n";
		}

		return Kmeans;
	}

	void center_to_cluster(double* Kmeans, int K, double* X, int sampleCount, int inputCountPerSample, double* colors) {
		double* Xn = new double[inputCountPerSample]; // Init with big int?
		double* Kn = new double[inputCountPerSample];
		double* distances = new double[sampleCount];
		for (int i = 0; i < sampleCount; i++)
			distances[i] = INT_MAX;

		for (int k = 0; k < K; k++)
		{
			// get one K
			for (int i = 0; i < inputCountPerSample; i++)
			{
				Kn[i] = Kmeans[(inputCountPerSample * k) + i];
			}
			
			// for Kn get all distances. If distance smaller, replace
			for (int n = 0; n < sampleCount; n++)
			{
				for (int i = 0; i < inputCountPerSample; i++)
				{
					Xn[i] = X[(inputCountPerSample * n) + i];
				}
				double distance = get_distanceK(Kn, Xn, inputCountPerSample);
				std::cout << distances[n] << " ";
				if (distance < distances[n])
				{
					distances[n] = distance;
					colors[n] = k;
				}
			}
			std::cout << "\n";
		}

	}
	void cluster_to_center(double* Kmeans, int K, double* Xtrain, int sampleCount, int inputCountperSample, double* old_Kmeans, double* colors) {
		// Store Kmeans in old_Kmeans
		old_Kmeans = Kmeans;

		double* KmeansTmp = new double[K * inputCountperSample];
		for (int i = 0; i < K * inputCountperSample; i++)
			KmeansTmp[i] = 0;

		double* color_occur = new double[sampleCount];
		for (int i = 0; i < sampleCount; i++)
			color_occur[i] = 0;


		for (int k = 0; k < K; k++)
		{
			for (int i = 0; i < sampleCount; i++) // accumulate coordinate
			{
				if (colors[i] == k) {
					color_occur[i] += 1;
					for (int n = 0; n < inputCountperSample; n++)
					{
						KmeansTmp[(k * inputCountperSample) + n] += Xtrain[(i * inputCountperSample) + n];
					}
				}
			}
		}
		for (int i = 0; i < K; i++) // make mean
		{
			for (int n = 0; n < inputCountperSample; n++)
			{
				KmeansTmp[(i * inputCountperSample) + n] /= color_occur[i];
				std::cout << KmeansTmp[(i * inputCountperSample) + n] << "\n";
			}

		}
		Kmeans = KmeansTmp;
	}

	bool should_stop(double* Kmeans, double* old_Kmeans, int size) {
		for (int i = 0; i < size; i++)
			if (Kmeans[i] != old_Kmeans[i])
				return false;
		return true;
	}

	SUPEREXPORT double* get_Kmeans(int K, double* Xtrain, int sampleCount, int inputCountperSample, int epochs) {
		double* Kmeans = select_random_k(K, Xtrain, sampleCount, inputCountperSample);
		double* old_Kmeans = new double[K * inputCountperSample];
		double* colors = new double[sampleCount * inputCountperSample];

		for (int i = 0; i < epochs; i++)
		{
			std::cout << i << " Epochs \n";
			center_to_cluster(Kmeans, K, Xtrain, sampleCount, inputCountperSample, colors);
			std::cout << " center_to_cluster \n";
			cluster_to_center(Kmeans, K, Xtrain, sampleCount, inputCountperSample, old_Kmeans, colors);
			std::cout << " cluster_to_center \n";

			if (should_stop(Kmeans, old_Kmeans, K * inputCountperSample))
			{
				std::cout << "stop \n";
				break;
			}
		}

		return Kmeans;
	}
}