#include "Kmeans.h"

extern "C" {

	double get_distance(double* Xpredict, double* Xn, int inputCountPerSample) {

		// l2-norm
		double accum = 0.;

		for (int i = 0; i < inputCountPerSample; ++i) {
			double res = Xpredict[i] - Xn[i];
			accum += res * res;
		}

		double norm = sqrt(accum);
		return norm;
	}

	double* select_random_k(int K, double* Xtrain, int sampleCount, int inputCountperSample) {
		double* Kmeans = new double[K * inputCountperSample];
		return Kmeans;
	}

	void center_to_cluster(double* Kmeans, int K, double* X, int sampleCount, int inputCountPerSample, double* colors) {
		double* Xn = new double[inputCountPerSample]; // Init with big int?
		double* Kn = new double[inputCountPerSample];
		double* distances = new double[inputCountPerSample];
		for (int i = 0; i < inputCountPerSample; i++)
			distances[i] = INT_MAX;

		for (int k = 0; k < K; k++)
		{
			// get one K
			for (int i = 0; i < inputCountPerSample; i++)
			{
				Kn[i] = Kmeans[(inputCountPerSample * k) + i];
			}
			// for one K get all distances. If distance smaller, replace
			for (int n = 0; n < sampleCount; n++)
			{
				for (int i = 0; i < inputCountPerSample; i++)
				{
					Xn[i] = X[(inputCountPerSample * n) + i];
				}
				double distance = get_distance(Kn, Xn, inputCountPerSample);
				if (distance < distances[n])
					colors[n] = k;
			}

		}


	}
	void cluster_to_center(double* Kmeans, int K, double* Xtrain, int sampleCount, int inputCountperSample, double* old_Kmeans, double* colors) {
		// Store Kmeans in old_Kmeans


		// For each K
		// Get all points from color K
		// Compute means of all K colors
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
			center_to_cluster(Kmeans, K, Xtrain, sampleCount, inputCountperSample, colors);
			cluster_to_center(Kmeans, K, Xtrain, sampleCount, inputCountperSample, old_Kmeans, colors);

			if (should_stop(Kmeans, old_Kmeans, inputCountperSample * sampleCount))
				break;
		}

		return Kmeans;
	}
}