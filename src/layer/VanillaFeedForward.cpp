/*
 * VanillaFeedForward.cpp
 *
 *  Created on: Dec 11, 2016
 *      Author: ken
 */

#include "VanillaFeedForward.h"

VanillaFeedForward::VanillaFeedForward(): mW(arma::Mat<double>()), mB(arma::Mat<double>()) {
	// TODO Auto-generated constructor stub

}

VanillaFeedForward::~VanillaFeedForward() {
	// TODO Auto-generated destructor stub
}

arma::Cube<double> VanillaFeedForward::feedForward(const arma::Cube<double>& z) {
	return arma::Cube<double>((const double*) (mW*arma::vectorise(z) + mB).begin(), 1, 1, z.size());
}

arma::field<arma::Cube<double>> VanillaFeedForward::feedForward(const arma::field<arma::Cube<double>>& zs) {
	arma::field<arma::Cube<double>> ys(zs.size());
	for (unsigned int i=0; i<zs.size(); ++i) {
		ys[i] = feedForward(zs[i]);
	}
	return ys;
}

arma::field<arma::Cube<double>> VanillaFeedForward::backProp(const arma::field<arma::Cube<double>>& deltas) {

}


