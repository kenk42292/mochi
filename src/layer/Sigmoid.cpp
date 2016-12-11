/*
 * Sigmoid.cpp
 *
 *  Created on: Dec 8, 2016
 *      Author: ken
 */

#include "Sigmoid.h"

Sigmoid::Sigmoid() {
	// TODO Auto-generated constructor stub

}

Sigmoid::~Sigmoid() {
	// TODO Auto-generated destructor stub
}

arma::field<arma::Cube<double>> Sigmoid::feedForward(const arma::field<arma::Cube<double>>& zs) {
	arma::field<arma::Cube<double>> ys(zs.size());
	arma::Cube<double> ones(zs[0].n_cols, zs[0].n_rows, zs[0].n_slices, arma::fill::ones);
	for (unsigned int i=0; i<zs.size(); ++i) {
		ys[i] = ones / (ones+arma::exp(-zs[i]));;
	}
	return ys;
}


arma::field<arma::Cube<double>> Sigmoid::backProp(const arma::field<arma::Cube<double>>& deltas) {
	arma::field<arma::Cube<double>> deltaPrevs(deltas.size());
	arma::Cube<double> ones(deltas[0].n_cols, deltas[0].n_rows, deltas[0].n_slices, arma::fill::ones);
	for (unsigned int i=0; i<deltas.size(); ++i) {
		deltaPrevs[i] = deltas[i] % ones / (ones+arma::exp(-deltas[i])) % (ones-ones / (ones+arma::exp(-deltas[i])));
	}
	return deltaPrevs;
}






