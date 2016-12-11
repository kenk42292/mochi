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

/* Single samples */
arma::Col<double> Layer::feedForward(const arma::Col<double>& z) {
	return z;
}
arma::Mat<double> Layer::feedForward(const arma::Mat<double>& z) {
	return z;
}
arma::Cube<double> Layer::feedForward(const arma::Cube<double>& z) {
	return z;
}
arma::Col<double> Layer::backProp(const arma::Col<double>& delta) {
	return delta;
}
arma::Mat<double> Layer::backProp(const arma::Mat<double>& delta) {
	return delta;
}
arma::Cube<double> Layer::backProp(const arma::Cube<double>& delta) {
	return delta;
}

/* Batch processing */
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





