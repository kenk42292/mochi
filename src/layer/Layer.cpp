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

std::vector<arma::Col<double>> Layer::feedForward(const std::vector<arma::Col<double>>& zs) {
	return zs;
}
std::vector<arma::Mat<double>> Layer::feedForward(const std::vector<arma::Mat<double>>& zs) {
	return zs;
}
std::vector<arma::Cube<double>> Layer::feedForward(const std::vector<arma::Cube<double>>& zs) {
	return zs;
}
std::vector<arma::Col<double>> Layer::backProp(const std::vector<arma::Col<double>>& deltas) {
	return deltas;
}
std::vector<arma::Mat<double>> Layer::backProp(const std::vector<arma::Mat<double>>& deltas) {
	return deltas;
}
std::vector<arma::Cube<double>> Layer::backProp(const std::vector<arma::Cube<double>>& deltas) {
	return deltas;
}





