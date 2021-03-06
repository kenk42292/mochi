/*
 * Softmax.cpp
 *
 *  Created on: Dec 11, 2016
 *      Author: ken
 */

#include "Softmax.hpp"

Softmax::Softmax() {}

Softmax::~Softmax() {}

arma::Cube<double> Softmax::softmax(const arma::Cube<double>& z) {
	arma::Cube<double> exp_z = arma::exp(z);
	return exp_z / arma::accu(exp_z);
}

arma::field<arma::Cube<double>> Softmax::feedForward(const arma::field<arma::Cube<double>>& xs) {
	arma::field<arma::Cube<double>> ys(xs.size());
	for (unsigned int i=0; i<xs.size(); ++i) {
		double m = fmax(0.0, xs[i].max());
		const arma::Cube<double>& xStable = xs[i]-m;
		arma::Cube<double> expX = arma::exp(xStable);
		ys[i] = expX / arma::accu(expX);
	}
	mYs = ys;
	return ys;
}

arma::field<arma::Cube<double>> Softmax::backProp(const arma::field<arma::Cube<double>>& deltas) {
	arma::field<arma::Cube<double>> dxs(deltas.size());
	for (unsigned int i=0; i<deltas.size(); ++i) {
		const arma::Col<double>& y = mYs[i];
		arma::Mat<double> jacobian = -y*y.t();
		jacobian.diag() = y%(1.0-y);
		arma::Col<double> delta = deltas[i];
		arma::Col<double> dx = jacobian*delta;
		dxs[i] = arma::Cube<double>((const double*) dx.begin(), 1, 1, dx.size());
	}
	return dxs;
}




