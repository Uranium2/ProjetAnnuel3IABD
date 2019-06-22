#include "Serial.h"

extern "C" {
	SUPEREXPORT void saveModel(void* model)
	{

		std::ofstream ofs("fifthgrade.ros", std::ios::binary);



		ofs.write((void*)& model, sizeof(model));
	}
}