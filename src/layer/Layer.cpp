/*
 * Layer.cpp
 *
 *  Created on: Dec 8, 2016
 *      Author: ken
 */

#include "Layer.hpp"

Layer::Layer() {}

Layer::~Layer() {}

arma::field<arma::Cube<double>> Layer::feedForward(const arma::field<arma::Cube<double>>& zs) {
	std::cout << "Layer base class feedforward called..." << std::endl;
	return zs;
}
arma::field<arma::Cube<double>> Layer::backProp(const arma::field<arma::Cube<double>>& deltas) {
	return deltas;
}





