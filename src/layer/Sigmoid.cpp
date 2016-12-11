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

/** Single Samples */

arma::Col<double> Sigmoid::feedForward(const arma::Col<double>& z) {
	arma::Col<double> ones(z.size(), arma::fill::ones);
	return ones / (ones+arma::exp(-z));
}

arma::Mat<double> Sigmoid::feedForward(const arma::Mat<double>& z) {
	arma::Col<double> ones(z.n_cols, z.n_rows, arma::fill::ones);
	return ones / (ones+arma::exp(-z));
}

arma::Cube<double> Sigmoid::feedForward(const arma::Cube<double>& z) {
	arma::Cube<double> ones(z.n_rows, z.n_cols, z.n_slices, arma::fill::ones);
	return ones / (ones+arma::exp(-z));
}

arma::Col<double> Sigmoid::backProp(const arma::Col<double>& delta) {
	arma::Col<double> ones(delta.size(), arma::fill::ones);
	return delta % feedForward(delta) % (ones-feedForward(delta));
}

arma::Mat<double> Sigmoid::backProp(const arma::Mat<double>& delta) {
	arma::Mat<double> ones(delta.n_cols, delta.n_rows, arma::fill::ones);
	return delta % feedForward(delta) % (ones-feedForward(delta));
}

arma::Cube<double> Sigmoid::backProp(const arma::Cube<double>& delta) {
	arma::Cube<double> ones(delta.n_rows, delta.n_cols, delta.n_slices, arma::fill::ones);
	return delta % feedForward(delta) % (ones-feedForward(delta));
}

/** Batch Processing */
std::vector<arma::Col<double>> Sigmoid::feedForward(const std::vector<arma::Col<double>>& zs) {
	std::vector<arma::Col<double>> ys(zs.size());
	arma::Col<double> ones(zs[0].size(), arma::fill::ones);
	for (unsigned int i=0; i<zs.size(); ++i) {
		ys[i] = ones / (ones+arma::exp(-zs[i]));;
	}
	return ys;
}

std::vector<arma::Mat<double>> Sigmoid::feedForward(const std::vector<arma::Mat<double>>& zs) {
	std::vector<arma::Mat<double>> ys(zs.size());
	arma::Mat<double> ones(zs[0].n_cols, zs[0].n_rows, arma::fill::ones);
	for (unsigned int i=0; i<zs.size(); ++i) {
		ys[i] = ones / (ones+arma::exp(-zs[i]));;
	}
	return ys;
}

std::vector<arma::Cube<double>> Sigmoid::feedForward(const std::vector<arma::Cube<double>>& zs) {
	std::vector<arma::Cube<double>> ys(zs.size());
	arma::Cube<double> ones(zs[0].n_cols, zs[0].n_rows, zs[0].n_slices, arma::fill::ones);
	for (unsigned int i=0; i<zs.size(); ++i) {
		ys[i] = ones / (ones+arma::exp(-zs[i]));;
	}
	return ys;
}

std::vector<arma::Col<double>> Sigmoid::backProp(const std::vector<arma::Col<double>>& deltas) {
	std::vector<arma::Col<double>> deltaPrevs(deltas.size());
	arma::Col<double> ones(deltas[0].size(), arma::fill::ones);
	for (unsigned int i=0; i<deltas.size(); ++i) {
		deltaPrevs[i] = deltas[i] % feedForward(deltas[i]) % (ones-feedForward(deltas[i]));
	}
	return deltaPrevs;
}

std::vector<arma::Mat<double>> Sigmoid::backProp(const std::vector<arma::Mat<double>>& deltas) {
	std::vector<arma::Mat<double>> deltaPrevs(deltas.size());
	arma::Mat<double> ones(deltas[0].n_cols, deltas[0].n_rows, arma::fill::ones);
	for (unsigned int i=0; i<deltas.size(); ++i) {
		deltaPrevs[i] = deltas[i] % feedForward(deltas[i]) % (ones-feedForward(deltas[i]));
	}
	return deltaPrevs;
}

std::vector<arma::Cube<double>> Sigmoid::backProp(const std::vector<arma::Cube<double>>& deltas) {
	std::vector<arma::Cube<double>> deltaPrevs(deltas.size());
	arma::Cube<double> ones(deltas[0].n_cols, deltas[0].n_rows, deltas[0].n_slices, arma::fill::ones);
	for (unsigned int i=0; i<deltas.size(); ++i) {
		deltaPrevs[i] = deltas[i] % feedForward(deltas[i]) % (ones-feedForward(deltas[i]));
	}
	return deltaPrevs;
}






