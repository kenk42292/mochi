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
double Quadratic::loss(arma::Cube<double> output, arma::Cube<double> y) {
	return 0.5*arma::accu(arma::square(output-y));
}

arma::Cube<double> Quadratic::loss_prime(const arma::Cube<double>& output, const arma::Cube<double>& y) {
	return output-y;
}

arma::field<arma::Cube<double>> Quadratic::loss_prime(const arma::field<arma::Cube<double>>& outputs, const arma::field<arma::Cube<double>>& ys) {
	arma::field<arma::Cube<double>> deltas(outputs.size()); //size may return num of all elements... rather, want length
	for (unsigned int i=0; i<deltas.size(); ++i) {
		deltas[i] = loss_prime(outputs[i], ys[i]);
	}
	return deltas;
}
