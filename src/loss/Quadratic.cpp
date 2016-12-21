/*
 * Quadratic.cpp
 *
 *  Created on: Dec 11, 2016
 *      Author: ken
 */

#include "Quadratic.hpp"

Quadratic::Quadratic() {
	// TODO Auto-generated constructor stub

}

Quadratic::~Quadratic() {
	// TODO Auto-generated destructor stub
}

double Quadratic::loss(arma::Cube<double> output, arma::Cube<double> y) {
	return 0.5*arma::accu(arma::square(output-y));
}

double Quadratic::loss(arma::field<arma::Cube<double>> outputs, arma::field<arma::Cube<double>> ys) {
	double total = 0.0;
	for (unsigned int i=0; i<outputs.size(); ++i) {
		total += loss(outputs[i], ys[i]);
	}
	return total;
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
