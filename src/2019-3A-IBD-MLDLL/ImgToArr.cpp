#include "ImgToArr.h"

double* buildYTrain(int nbImg, int type) {
	double* res = new double[nbImg * 3];
	int i;
	int max;
	for (i = 0; i < nbImg * 3; i++)
		res[i] = -1.0;

	switch (type) {
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

double* loadImgToPredict(char* path, int w, int h)
{
	int index = 0;
	double* res = new double[w * h];
	folderToArr(path, w, h, 1, &index, res);
	return res;
}


double* buildXTrain(char* pathFolderFPS, char* pathFolderRTS, char* pathFolderMOBA, int w, int h, int nbImg) {
	int index = 0;
	double* imgs = new double[nbImg * 3 * w * h];
	imgs[10] == 1.0;
	folderToArr(pathFolderFPS, w, h, nbImg, &index, imgs);
	folderToArr(pathFolderRTS, w, h, nbImg, &index, imgs);
	folderToArr(pathFolderMOBA, w, h, nbImg, &index, imgs);

	return imgs;
}

void folderToArr(char* pathFolder, int w, int h, int nbImg, int* index, double* imgs) {
	int pos = 0;
	std::string path = pathFolder;
	for (const auto& entry : std::filesystem::directory_iterator(path))
	{
		if (pos >= nbImg)
			break;
		std::string path_string = entry.path().u8string();
		pos++;
		pathToArr(path_string, w, h, index, imgs);

	}
}

void pathToArr(std::string path, int w, int h, int* index, double* imgs) {
	cv::Mat image = cv::imread(path);
	imgToArr(image, w, h, index, imgs);
	image.release();
}

void imgToArr(cv::Mat image, int w, int h, int* index, double* imgs) {
	cv::Size size(w, h);
	cv::resize(image, image, size);
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			int grayscale = ((int)image.at<cv::Vec3b>(i, j)[2] + (int)image.at<cv::Vec3b>(i, j)[1] + (int)image.at<cv::Vec3b>(i, j)[0]) / 3;
			double normGrayscale = (double)(grayscale - 0) / (255 - 0);
			imgs[*index] = normGrayscale;
			*index = *index + 1;
			//std::cout << "R: " << (int)image.at<cv::Vec3b>(i, j)[2] << " G: " << (int)image.at<cv::Vec3b>(i, j)[1] << " B: " << (int)image.at<cv::Vec3b>(i, j)[0] << "\n";
			//std::cout << "pixel pos " << i << " " << arr.at(i) << "\n";
		}
		//std::cout << "\n";
	}
}