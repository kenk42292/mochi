/*
 * CrossEntropy.cpp
 *
 *  Created on: Dec 11, 2016
 *      Author: ken
 */

#include "CrossEntropy.hpp"

CrossEntropy::CrossEntropy() {}

CrossEntropy::~CrossEntropy() {}

double CrossEntropy::loss(arma::Cube<double> output, arma::Cube<double> y) {
	const arma::Col<double>& outputVector = output;
	const arma::Col<double>& yVector = y;
	return -arma::accu(y%arma::log(output) + (1.0-y)%arma::log(1.0-output));
}

arma::Cube<double> CrossEntropy::loss_prime(const arma::Cube<double>& output, const arma::Cube<double>& y) {
	return -y/output;
}

arma::field<arma::Cube<double>> CrossEntropy::loss_prime(const arma::field<arma::Cube<double>>& outputs, const arma::field<arma::Cube<double>>& ys) {
	arma::field<arma::Cube<double>> deltas(outputs.size());
	for (unsigned int i=0; i<outputs.size(); ++i) {
		deltas[i] = loss_prime(outputs[i], ys[i]);
	}
	return deltas;
}

