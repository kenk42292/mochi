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
	return -arma::accu(y%arma::log(output) + (1.0-y)%arma::log(1.0-output));
}

arma::Cube<double> CrossEntropy::loss_prime(const arma::Cube<double>& output, const arma::Cube<double>& y) {
	return -y/output;
}

