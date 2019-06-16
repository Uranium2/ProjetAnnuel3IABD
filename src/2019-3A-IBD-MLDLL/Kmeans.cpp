#include "Kmeans.h"

extern "C" {

	double* select_random_k(int K, double* Xtrain, int sampleCount, int inputCountperSample) {
		double* Kmeans = new double[K * inputCountperSample];
		return Kmeans;
	}

	void center_to_cluster(double* Kmeans, int K, double* Xtrain, int sampleCount, int inputCountperSample) {
	}
	void cluster_to_center(double* Kmeans, int K, double* Xtrain, int sampleCount, int inputCountperSample, double* old_Kmeans) {
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

		for (int i = 0; i < epochs; i++)
		{
			center_to_cluster(Kmeans, K, Xtrain, sampleCount, inputCountperSample);
			cluster_to_center(Kmeans, K, Xtrain, sampleCount, inputCountperSample, old_Kmeans);

			if (should_stop(Kmeans, old_Kmeans, inputCountperSample * sampleCount))
				break;
		}

		return Kmeans;
	}
}