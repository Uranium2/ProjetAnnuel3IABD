#include "Linear.h"

extern "C" {
	SUPEREXPORT double* create_linear_model(int inputCountPerSample) {
		double* W = new double[inputCountPerSample + 1];
		double low = -1.0;
		double up = 1.0;
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::default_random_engine generator(seed);

		std::uniform_real_distribution<double> distribution(low, up);
		for (int i = 0; i < inputCountPerSample + 1; i++)
		{
			W[i] = distribution(generator);
		}
		return W;
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
	double feedForward(double* X, double* W, int inputCountPerSample)
	{
		double result = 0;
		for (int i = 0; i < inputCountPerSample; i++)
		{
			result += X[i] * W[i];
		}
		return std::tanh(result);
	}

	SUPEREXPORT double predict_classification_rosenblatt(double* W, double* X, int inputCountPerSample)
	{
		double* Xnew = new double[inputCountPerSample + 1];
		Xnew[0] = 1;
		int pos = 0;
		for (int i = 1; i < inputCountPerSample + 1; i++)
			Xnew[i] = X[pos++];

		double result = 0;
		for (int i = 0; i < inputCountPerSample + 1; i++)
			result += Xnew[i] * W[i];

		auto res = std::tanh(result);

		if (res >= 0)
			return 1;
		return -1;
	}

	SUPEREXPORT void fit_classification_rosenblatt_rule(
		double* W,
		double* XTrain,
		int sampleCount,
		int inputCountPerSample,
		double* YTrain,
		double alpha,
		int epochs)
	{
		double output = 0.0;
		double* Xactual = new double[inputCountPerSample + 1];
		Xactual[0] = 1;
		for (int e = 1; e < epochs + 1; e++)
		{
			double* Xout = new double[(double)sampleCount];
			int pos = 0;
			for (int img = 0; img < sampleCount; img++)
			{
				for (int input = 1; input < inputCountPerSample + 1; input++)
					Xactual[input] = XTrain[pos++];

				output = feedForward(Xactual, W, inputCountPerSample + 1);
				Xout[img] = output;

				for (int i = 0; i < inputCountPerSample + 1; i++)
					W[i] = W[i] + alpha * (YTrain[img] - output) * Xactual[i];

			}
			double loss = mse_loss(YTrain, Xout, sampleCount);
			if (e % 10 == 0 || e == epochs - 1)
				printf("Epoch: %d loss: %f\n", e, loss);
		}
	}

	SUPEREXPORT double* fit_regression(
		double* XTrain,
		int sampleCount,
		int inputCountPerSample,
		double* YTrain
	)
	{
		Eigen::MatrixXd X(sampleCount, inputCountPerSample + 1);
		Eigen::MatrixXd Y(sampleCount, 1);

		int pos = 0;
		for (int x = 0; x < sampleCount; x++)
		{
			for (int y = 0; y < inputCountPerSample + 1; y++)
			{
				if (y == 0)
					X(x, y) = 1;
				else
				{
					X(x, y) = XTrain[pos++];
					if (x == y)
						X(x, y) += 0.00001;
				}
			}
		}

		for (int x = 0; x < sampleCount; x++)
			Y(x, 0) = YTrain[x];


		Eigen::MatrixXd W(inputCountPerSample + 1, 1);
		Eigen::MatrixXd transposeX = X.transpose();
		Eigen::MatrixXd multX = transposeX * X;
		Eigen::MatrixXd pseudo_inverse = multX.completeOrthogonalDecomposition().pseudoInverse();
		Eigen::MatrixXd mult_inv_trans = pseudo_inverse * transposeX;
		W = mult_inv_trans * Y;


		double* Wmat = new double[inputCountPerSample + 1];

		for (int i = 0; i < inputCountPerSample + 1; i++)
			Wmat[i] = W(i);

		return Wmat;
	}

	SUPEREXPORT double predict_regression(double* W, double* X, int inputCountPerSample)
	{
		double* Xnew = new double[inputCountPerSample + 1];
		Xnew[0] = 1;
		int pos = 0;
		for (int i = 1; i < inputCountPerSample + 1; i++)
			Xnew[i] = X[pos++];

		double result = 0;
		for (int i = 0; i < inputCountPerSample + 1; i++)
			result += Xnew[i] * W[i];

		return result;
	}
}