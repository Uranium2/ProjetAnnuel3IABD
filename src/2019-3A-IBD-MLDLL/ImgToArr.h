#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <filesystem>

double* buildYTrain(int nbImg, int type);

double* buildXTrain(char* pathFolderFPS, char* pathFolderRTS,char* pathFolderMOBA, int w, int h, int nbImg);

double* loadImgToPredict(char* path, int w, int h);

void folderToArr(char* pathFolder, int w, int h, int nbImg, int* index, double* imgs);

void pathToArr(std::string path, int w, int h, int* index, double* imgs);

void imgToArr(cv::Mat image, int w, int h, int* index, double* imgs);