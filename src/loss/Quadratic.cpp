/*
 * Quadratic.cpp
 *
 *  Created on: Dec 11, 2016
 *      Author: ken
 */

#include "Quadratic.h"

Quadratic::Quadratic() {
	// TODO Auto-generated constructor stub

}

Quadratic::~Quadratic() {
	// TODO Auto-generated destructor stub
}

/** Single Sample Loss*/
double Quadratic::loss(arma::Col<double> output, arma::Col<double> y) {
	return 0.5*arma::accu(arma::square(output-y));
}

arma::Col<double> Quadratic::loss_prime(arma::Col<double> output, arma::Col<double> y) {
	return output-y;
}
