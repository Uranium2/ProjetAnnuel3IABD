#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <filesystem>

double* buildYTrain(int nbImg, int type);

double* buildXTrain(char* pathFolderFPS, char* pathFolderRTS,char* pathFolderMOBA, int w, int h, int nbImg);

std::vector< double > folderToArr(char* pathFolder, int w, int h, int nbImg);

std::vector< double > pathToArr(std::string path, int w, int h);

std::vector< double > imgToArr(cv::Mat image, int w, int h);