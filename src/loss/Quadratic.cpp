/*
 * Quadratic.cpp
 *
 *  Created on: Dec 11, 2016
 *      Author: ken
 */

#include "Quadratic.hpp"

Quadratic::Quadratic() {}

Quadratic::~Quadratic() {}

double Quadratic::loss(arma::Cube<double> output, arma::Cube<double> y) {
	return 0.5*arma::accu(arma::square(output-y));
}

arma::Cube<double> Quadratic::loss_prime(const arma::Cube<double>& output, const arma::Cube<double>& y) {
	return output-y;
}

