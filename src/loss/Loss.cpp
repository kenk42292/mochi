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

/** Single Sample Loss*/
double Loss::loss(arma::Col<double> output, arma::Col<double> y) {
//	throw std::exception("Not Implemented");
	return 0.0;
}

arma::Col<double> Loss::loss_prime(arma::Col<double> output, arma::Col<double> y) {
//	throw std::exception("Not Implemented");
	return arma::Col<double>(1, arma::fill::zeros);
}

/** Batch Loss */
std::vector<double> Loss::loss(std::vector<arma::Col<double>> outputs, std::vector<arma::Col<double>> ys) {
	std::vector<double> losses(outputs.size());
	for (unsigned int i=0; i<outputs.size(); ++i) {
		losses[i] = loss(outputs[i], ys[i]);
	}
	return losses;
}

std::vector<arma::Col<double>> Loss::loss_prime(std::vector<arma::Col<double>> outputs, std::vector<arma::Col<double>> ys) {
	std::vector<arma::Col<double>> deltas(outputs.size());
	for (unsigned int i=0; i<outputs.size(); ++i) {
		deltas[i] = loss_prime(outputs[i], ys[i]);
	}
	return deltas;
}

/** Total Batch Loss */
double Loss::totalLoss(std::vector<arma::Col<double>> outputs,
		std::vector<arma::Col<double>> ys) {
	double totalLoss = 0;
	for (unsigned int i = 0; i < outputs.size(); ++i) {
		totalLoss += loss(outputs[i], ys[i]);
	}
	return totalLoss;
}
