/*
 * VanillaFeedForward.cpp
 *
 *  Created on: Dec 11, 2016
 *      Author: ken
 */

#include "VanillaFeedForward.h"

VanillaFeedForward::VanillaFeedForward(unsigned int nIn, unsigned int nOut) :
		mW(arma::Mat<double>(nOut, nIn, arma::fill::randu)),
		mB(arma::Col<double>(nOut, arma::fill::randu)) {
	// TODO Auto-generated constructor stub


}

VanillaFeedForward::~VanillaFeedForward() {
	// TODO Auto-generated destructor stub
}

arma::Cube<double> VanillaFeedForward::feedForward(
		const arma::Cube<double>& x) {
	arma::Col<double> v = arma::vectorise(x);
	v = arma::vectorise(mW * v) + mB;
	return arma::Cube<double>((const double*) v.begin(), 1, 1, v.size());
}

arma::field<arma::Cube<double>> VanillaFeedForward::feedForward(
		const arma::field<arma::Cube<double>>& xs) {
	mxs = xs;
	arma::field<arma::Cube<double>> ys(xs.size());
	for (unsigned int i = 0; i < xs.size(); ++i) {
		ys[i] = feedForward(xs[i]);
	}
	return ys;
}

arma::field<arma::Cube<double>> VanillaFeedForward::backProp(
		const arma::field<arma::Cube<double>>& deltas) {
	arma::Mat<double> dw(mW.n_rows, mW.n_cols, arma::fill::zeros);
	arma::Col<double> db(mB.n_elem, arma::fill::zeros);
	arma::field<arma::Cube<double>> dxs(deltas.size());
	for (unsigned int i=0; i<deltas.size(); ++i) {
		dw += arma::Mat<double>(deltas[i].begin(), deltas[i].size(), 1)*arma::Mat<double>(mxs[i].begin(), 1, mxs[i].size());
		db += arma::vectorise(deltas[i]);
		arma::Mat<double> dx = mW.t()*arma::vectorise(deltas[i]);
		dxs[i] = arma::Cube<double>(dx.begin(), 1, 1, dx.size()); // do I need to vectorise the delta...?
	}
	//TODO: Don't hard-code etas... and make optimizer programmatic
	mW -= 0.03*dw/deltas.size();
	mB -= 0.03*db/deltas.size();
	return dxs;
}

