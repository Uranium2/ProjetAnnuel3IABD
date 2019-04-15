#include "ImgToArr.h"

double* buildXTrain(char* pathFolderFPS, char* pathFolderRTS,char* pathFolderMOBA, int w, int h, int nbImg){
	std::vector< double > XTrainFPS = folderToArr(pathFolderFPS, w, h, nbImg);
	std::vector< double > XTrainRTS = folderToArr(pathFolderRTS, w, h, nbImg);
	std::vector< double > XTrainMOBA = folderToArr(pathFolderMOBA, w, h, nbImg);
	std::vector< double > arr;

	arr.reserve(arr.size() + XTrainFPS.size());
	arr.insert(arr.end(), XTrainFPS.begin(), XTrainFPS.end());
	arr.reserve(arr.size() + XTrainRTS.size());
	arr.insert(arr.end(), XTrainRTS.begin(), XTrainRTS.end());
	arr.reserve(arr.size() + XTrainMOBA.size());
	arr.insert(arr.end(), XTrainMOBA.begin(), XTrainMOBA.end());

	double* res = &arr[0]; // mem is continious in vector, so first pointer gets all array
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
	resize(image, image, size);
	std::vector< double > arr;
	int pos = 0;
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			arr.push_back(((int)image.at<cv::Vec3b>(i, j)[0] + (int)image.at<cv::Vec3b>(i, j)[1] + (int)image.at<cv::Vec3b>(i, j)[2]) / 3);
			//std::cout << "R: " << (int)image.at<cv::Vec3b>(i, j)[0] << " G: " << (int)image.at<cv::Vec3b>(i, j)[1] << " B: " << (int)image.at<cv::Vec3b>(i, j)[2] << " ";
			pos++;
		}
		//std::cout << "\n";
	}
	return arr;
}