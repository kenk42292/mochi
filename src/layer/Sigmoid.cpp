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

arma::Col<double> Sigmoid::feedForward(arma::Col<double> z) {
	arma::Col<double> ones(z.size(), arma::fill::ones);
	return ones / (ones+arma::exp(-z));
}

arma::Mat<double> Sigmoid::feedForward(arma::Mat<double> z) {
	arma::Col<double> ones(z.n_cols, z.n_rows, arma::fill::ones);
	return ones / (ones+arma::exp(-z));
}

arma::Cube<double> Sigmoid::feedForward(arma::Cube<double> z) {
	arma::Cube<double> ones(z.n_rows, z.n_cols, z.n_slices, arma::fill::ones);
	return ones / (ones+arma::exp(-z));
}

