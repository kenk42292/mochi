/*
 * LossFactory.cpp
 *
 *  Created on: Jan 7, 2017
 *      Author: ken
 */

#include "LossFactory.hpp"

LossFactory::LossFactory() {}

LossFactory::~LossFactory() {}

Loss* LossFactory::createLoss(Configuration conf) {
	Loss* loss;
	std::map<std::string, std::string> lossConfig = conf.lossConfig();
	std::string lossType = lossConfig["type"];
	/* Find possible regularization */
	bool useReg = false;

	// Should define, AND inject reg function into constructor of loss
	if (lossConfig.count("reg")==1) {

	}

	if (lossType.compare("quadratic") == 0) {
		std::cout << "Setting to Quadratic Loss function" << std::endl;
		if (lossConfig.count("reg")==1)
		return new Quadratic();
	} else if (lossType.compare("crossentropy") == 0) {
		std::cout << "Setting to Cross Entropy Loss function" << std::endl;
		return new CrossEntropy();
	}
	return new CrossEntropy();
}

