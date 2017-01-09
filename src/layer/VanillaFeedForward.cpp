/*
 * VanillaFeedForward.cpp
 *
 *  Created on: Dec 11, 2016
 *      Author: ken
 */

#include "VanillaFeedForward.hpp"
#include <math.h>

VanillaFeedForward::VanillaFeedForward(unsigned int batchSize, unsigned int nIn, unsigned int nOut,
		Optimizer* optimizer, double wdecay) :
		mBatchSize(batchSize),
		mW(arma::Cube<double>(nOut, nIn, 1, arma::fill::randn)), mB(
				arma::Cube<double>(1, 1, nOut, arma::fill::zeros)),
				mWdecay(wdecay) {
	mW /= sqrt(nIn);
	mOptimizer = optimizer;
}

VanillaFeedForward::~VanillaFeedForward() {
	delete mOptimizer;
}

arma::Cube<double> VanillaFeedForward::feedForward(
		const arma::Cube<double>& x) {
	const arma::Mat<double>& weightVector = mW.slice(0);
	const arma::Col<double>& xVector = arma::Col<double>(
			(const double*) x.begin(), x.size());
	;
	const arma::Col<double>& biasVector = mB;
	arma::Col<double> v = weightVector * xVector + biasVector;
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

arma::field<arma::Cube<double>> VanillaFeedForward::getGrads(const arma::field<arma::Cube<double>>& deltas) {
	arma::Mat<double> dw(mW.n_rows, mW.n_cols, arma::fill::zeros);
	arma::Col<double> db(mB.n_elem, arma::fill::zeros);
	arma::field<arma::Cube<double>> grads(2+deltas.size());
	for (unsigned int i = 0; i < deltas.size(); ++i) {
		const arma::Col<double>& p = deltas[i];
		const arma::Col<double>& q = arma::Col<double>(
				(const double*) mxs[i].begin(),
				mxs[i].size());
		dw += p * q.t();
		db += p;
		const arma::Mat<double>& dx = mW.slice(0).t() * p;
		grads[i+2] = arma::Cube<double>(dx.begin(), 1, 1, dx.size());
	}
	grads[0] = arma::Cube<double>((const double*) dw.begin(), mW.n_rows,
			mW.n_cols, 1) + mWdecay*mW;
	grads[1] = arma::Cube<double>((const double*) db.begin(), 1, 1, db.size());
	return grads;
}

arma::field<arma::Cube<double>> VanillaFeedForward::backProp(
		const arma::field<arma::Cube<double>>& deltas) {

	arma::field<arma::Cube<double>> grads = getGrads(deltas);
	const arma::field<arma::Cube<double>>& paramChange = mOptimizer->delta(
			grads.rows(0, 1), deltas.size());
	mW -= paramChange[0];
	mB -= paramChange[1];

	return grads.rows(2, grads.size()-1);
}

