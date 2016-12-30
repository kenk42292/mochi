/*
 * Sigmoid.cpp
 *
 *  Created on: Dec 8, 2016
 *      Author: ken
 */

#include "Sigmoid.hpp"

Sigmoid::Sigmoid() {
}

Sigmoid::~Sigmoid() {
}

arma::field<arma::Cube<double>> Sigmoid::feedForward(
		const arma::field<arma::Cube<double>>& xs) {
	arma::field<arma::Cube<double>> ys(xs.size());
	for (unsigned int i = 0; i < xs.size(); ++i) {
		ys[i] = 1.0 / (1.0 + arma::exp(-xs[i]));
		;
	}
	mYs = ys;
	return ys;
}

arma::field<arma::Cube<double>> Sigmoid::backProp(
		const arma::field<arma::Cube<double>>& deltas) {
	arma::field<arma::Cube<double>> deltaPrevs(deltas.size());
	for (unsigned int i = 0; i < deltas.size(); ++i) {
		//TODO: Make dim. checking btwn cubes a utils function
		if (deltas[i].n_slices != mYs[i].n_slices
				|| deltas[i].n_rows != mYs[i].n_rows
				|| deltas[i].n_cols != mYs[i].n_cols) {
			deltaPrevs[i] = arma::Cube<double>(deltas[i].begin(), mYs[i].n_rows,
					mYs[i].n_cols, mYs[i].n_slices) % mYs[i] % (1.0 - mYs[i]);
		} else {
			deltaPrevs[i] = deltas[i] % mYs[i] % (1.0 - mYs[i]);
		}
	}
	return deltaPrevs;
}

