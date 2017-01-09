/*
 * Loss.cpp
 *
 *  Created on: Dec 11, 2016
 *      Author: ken
 */

#include "Loss.hpp"

Loss::Loss() {}

Loss::~Loss() {}

double Loss::loss(arma::Cube<double> output, arma::Cube<double> y) {
	return 0.0;
}

double Loss::loss(arma::field<arma::Cube<double>> outputs, arma::field<arma::Cube<double>> ys) {
	double total = 0.0;
	for (unsigned int i=0; i<outputs.size(); ++i) {
		total += loss(outputs[i], ys[i]);
	}
	return total;
}

arma::Cube<double> Loss::loss_prime(const arma::Cube<double>& output,
		const arma::Cube<double>& y) {
	return arma::Cube<double>(1, 1, 1, arma::fill::zeros);
}

arma::field<arma::Cube<double>> Loss::loss_prime(
		const arma::field<arma::Cube<double>>& outputs,
		const arma::field<arma::Cube<double>>& ys) {
	arma::field<arma::Cube<double>> deltas(outputs.size());
	for (unsigned int i = 0; i < outputs.size(); ++i) {
		deltas[i] = loss_prime(outputs[i], ys[i]);
	}
	return deltas;
}
