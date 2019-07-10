#pragma once
#if _WIN32
#define SUPEREXPORT __declspec(dllexport)
#else
#define SUPEREXPORT 
#endif
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/QR>

#include <iostream>