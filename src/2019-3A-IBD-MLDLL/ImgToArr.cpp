#include "ImgToArr.h"
double* folderToArr(char* pathFolder, int w, int h, int nbImg) {
	std::vector< double > arr;
	int pos = 0;
	std::string path = pathFolder;

	for (const auto& entry : std::filesystem::directory_iterator(path))
	{

		std::string path_string = entry.path().u8string();
		std::cout << "POS: " << pos << " " << path_string << std::endl;
		std::vector< double > tmp = pathToArr(path_string, w, h);

		arr.reserve(arr.size() + tmp.size()); // preallocate memory
		arr.insert(arr.end(), arr.begin(), arr.end());
		arr.insert(arr.end(), tmp.begin(), tmp.end());
		pos++;

	}
	double* res = &arr[0]; // mem is continious in vector, so first pointer gets all array
	return res;
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