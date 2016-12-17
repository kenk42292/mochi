/*
 * Utils.cpp
 *
 *  Created on: Dec 16, 2016
 *      Author: ken
 */

#include "Utils.hpp"

Utils::Utils() {
	// TODO Auto-generated constructor stub

}

Utils::~Utils() {
	// TODO Auto-generated destructor stub
}

void Utils::shuffle(arma::field<arma::Cube<double>>& inputs, arma::field<arma::Cube<double>>& outputs) {
	unsigned int swapIndex;
	unsigned int n = inputs.size();
	for (unsigned int i=0; i<inputs.size(); ++i) {
		swapIndex = i + (rand()*(int)(n-1-i)/RAND_MAX);
		arma::Cube<double>& tempIn = inputs[i];
		arma::Cube<double>& tempOut = outputs[i];
		inputs[i] = inputs[swapIndex];
		outputs[i] = outputs[swapIndex];
		inputs[swapIndex] = tempIn;
		outputs[swapIndex] = tempOut;
	}
}

void Utils::printConfig(std::vector<std::map<std::string, std::string>> layersConfig) {
	for (auto m : layersConfig) {
		std::cout << "LAYER---------------------------------------" << std::endl;
		for (auto const& pair : m) {
			std::cout << "\t\t" << pair.first << " : " << pair.second << std::endl;
		}
	}
}
