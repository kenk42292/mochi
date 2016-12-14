/*
 * Loss.cpp
 *
 *  Created on: Dec 11, 2016
 *      Author: ken
 */

#include "Loss.h"

Loss::Loss() {
	// TODO Auto-generated constructor stub

}

Loss::~Loss() {
	// TODO Auto-generated destructor stub
}

double Loss::loss(arma::Cube<double> output, arma::Cube<double> y) {
	return 0.0;
}

arma::Cube<double> Loss::loss_prime(const arma::Cube<double>& output,
		const arma::Cube<double>& y) {
	return arma::Cube<double>(1, 1, 1, arma::fill::zeros);
}

arma::field<arma::Cube<double>> Loss::loss_prime(
		const arma::field<arma::Cube<double>>& outputs,
		const arma::field<arma::Cube<double>>& ys) {
	arma::field<arma::Cube<double>> result(outputs.size());
	for (unsigned int i=0; i<outputs.size(); ++i) {
		result[i] = loss_prime(outputs[i], ys[i]);
	}
	return result;
}

