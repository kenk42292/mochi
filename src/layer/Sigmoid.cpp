/*
 * Sigmoid.cpp
 *
 *  Created on: Dec 8, 2016
 *      Author: ken
 */

#include "Sigmoid.hpp"

Sigmoid::Sigmoid() {}

Sigmoid::~Sigmoid() {}

arma::field<arma::Cube<double>> Sigmoid::feedForward(const arma::field<arma::Cube<double>>& xs) {
	arma::field<arma::Cube<double>> ys(xs.size());
	for (unsigned int i=0; i<xs.size(); ++i) {
		ys[i] = 1.0 / (1.0+arma::exp(-xs[i]));;
	}
	mYs = ys;
	return ys;
}


arma::field<arma::Cube<double>> Sigmoid::backProp(const arma::field<arma::Cube<double>>& deltas) {
	arma::field<arma::Cube<double>> deltaPrevs(deltas.size());
	for (unsigned int i=0; i<deltas.size(); ++i) {
		deltaPrevs[i] = deltas[i] % mYs[i] % (1.0-mYs[i]);
	}
	return deltaPrevs;
}






