/*
 * Layer.cpp
 *
 *  Created on: Dec 8, 2016
 *      Author: ken
 */

#include "Layer.h"

Layer::Layer() {
	// TODO Auto-generated constructor stub

}

Layer::~Layer() {
	// TODO Auto-generated destructor stub
}

arma::field<arma::Cube<double>> Layer::feedForward(const arma::field<arma::Cube<double>>& zs) {
	std::cout << "Layer base class feedforward called..." << std::endl;
	return zs;
}
arma::field<arma::Cube<double>> Layer::backProp(const arma::field<arma::Cube<double>>& deltas) {
	return deltas;
}





