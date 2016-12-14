/*
 * Softmax.cpp
 *
 *  Created on: Dec 11, 2016
 *      Author: ken
 */

#include "Softmax.h"

Softmax::Softmax() {
	// TODO Auto-generated constructor stub

}

Softmax::~Softmax() {
	// TODO Auto-generated destructor stub
}

arma::Cube<double> Softmax::softmax(const arma::Cube<double>& z) {
	arma::Cube<double> exp_z = arma::exp(z);
	return exp_z / arma::accu(exp_z);
}

arma::field<arma::Cube<double>> Softmax::feedForward(const arma::field<arma::Cube<double>>& zs) {
	arma::field<arma::Cube<double>> ys(zs.size());
	for (unsigned int i=0; i<zs.size(); ++i) {
		ys[i] = softmax(zs[i]);
	}
	return ys;
}

arma::field<arma::Cube<double>> Softmax::backProp(const arma::field<arma::Cube<double>>& deltas) {

}




