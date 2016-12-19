/*
 * Softplus.cpp
 *
 *  Created on: Dec 11, 2016
 *      Author: ken
 */

#include "Softplus.hpp"

Softplus::Softplus() {}

Softplus::~Softplus() {}


arma::field<arma::Cube<double>> Softplus::feedForward(const arma::field<arma::Cube<double>>& xs) {
	mxs = xs;
	arma::field<arma::Cube<double>> ys(xs.size());
	for (unsigned int i=0; i<xs.size(); ++i) {
		ys[i] = arma::log(1.0+arma::exp(xs[i]));
	}
	return ys;
}

arma::field<arma::Cube<double>> Softplus::backProp(const arma::field<arma::Cube<double>>& deltas) {
	arma::field<arma::Cube<double>> dxs(deltas.size());
	for (unsigned int i=0; i<deltas.size(); ++i) {
		dxs[i] = deltas[i] % (1.0 / (1.0+arma::exp(-mxs[i])));
	}
	return dxs;
}
