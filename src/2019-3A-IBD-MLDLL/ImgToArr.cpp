#include "ImgToArr.h"

double* buildYTrain(int nbImg, int type) {
	double* res = new double[nbImg * 3];
	int i;
	int max;
	for (i = 0; i < nbImg * 3; i++)
		res[i] = -1.0;

	switch(type) {
		case (1):
				i = 0;
				max = nbImg;
				break;
		case (2):
				i = nbImg;
				max = nbImg * 2;
				break;
		case (3):
				i = nbImg * 2;
				max = nbImg * 3;
				break;
		default:
				i = 0;
				max = nbImg;
				break;
	}

	for (; i < max; i++)
		res[i] = 1.0;

	return res;
}

double* buildXTrain(char* pathFolderFPS, char* pathFolderRTS,char* pathFolderMOBA, int w, int h, int nbImg){
	std::vector< double > XTrainFPS = folderToArr(pathFolderFPS, w, h, nbImg);
	std::vector< double > XTrainRTS = folderToArr(pathFolderRTS, w, h, nbImg);
	std::vector< double > XTrainMOBA = folderToArr(pathFolderMOBA, w, h, nbImg);

	double* res = new double[nbImg * 3 * w * h];

	int pos = 0;
	for (int i = 0; i < XTrainFPS.size(); i++) {
		res[pos] = XTrainFPS.at(i);
		pos++;
	}
	for (int i = 0; i < XTrainRTS.size(); i++) {
		res[pos] = XTrainRTS.at(i);
		pos++;
	}
	for (int i = 0; i < XTrainMOBA.size(); i++) {
		res[pos] = XTrainMOBA.at(i);
		pos++;
	}
	return res;
}

std::vector< double > folderToArr(char* pathFolder, int w, int h, int nbImg) {
	std::vector< double > arr;
	int pos = 0;
	std::string path = pathFolder;
	for (const auto& entry : std::filesystem::directory_iterator(path))
	{
		if (pos >= nbImg)
			break;
		std::string path_string = entry.path().u8string();

		std::vector< double > tmp = pathToArr(path_string, w, h);

		arr.reserve(arr.size() + tmp.size()); // preallocate memory
		arr.insert(arr.end(), tmp.begin(), tmp.end());

		pos++;

	}
	return arr;
}

std::vector< double > pathToArr(std::string path, int w, int h) {
	cv::Mat image = cv::imread(path);
	std::vector< double > res = imgToArr(image, w, h);
	image.release();
	return res;
}

std::vector< double > imgToArr(cv::Mat image, int w, int h) {
	cv::Size size(w, h);
	cv::resize(image, image, size);
	std::vector< double > arr;
	int pos = 0;
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			int grayscale = ((int)image.at<cv::Vec3b>(i, j)[2] + (int)image.at<cv::Vec3b>(i, j)[1] + (int)image.at<cv::Vec3b>(i, j)[0]) / 3;
			double normGrayscale = (double)(grayscale - 0) / (255 - 0);
			arr.push_back(normGrayscale);
			//std::cout << "R: " << (int)image.at<cv::Vec3b>(i, j)[2] << " G: " << (int)image.at<cv::Vec3b>(i, j)[1] << " B: " << (int)image.at<cv::Vec3b>(i, j)[0] << "\n";
			//std::cout << "pixel pos " << i << " " << arr.at(i) << "\n";
			pos++;
		}
		//std::cout << "\n";
	}
	return arr;
}