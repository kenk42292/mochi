/*
 * VanillaFeedForward.h
 *
 *  Created on: Dec 11, 2016
 *      Author: ken
 */

#ifndef LAYER_VANILLAFEEDFORWARD_HPP_
#define LAYER_VANILLAFEEDFORWARD_HPP_

#include "Layer.hpp"

class VanillaFeedForward: public Layer {
private:
	unsigned int mBatchSize;
	arma::Cube<double> mW;
	arma::Cube<double> mB;
	/** Stored batch inputs for back propagation */
	arma::field<arma::Cube<double>> mxs;
	Optimizer* mOptimizer;
	double mWdecay;
	friend class VanillaFeedForwardTest;
public:
	VanillaFeedForward(unsigned int batchSize, unsigned int nIn, unsigned int nOut, Optimizer* optimizer, double wdecay);
	virtual ~VanillaFeedForward();

	arma::Cube<double> feedForward(const arma::Cube<double>& x);
	arma::field<arma::Cube<double>> feedForward(const arma::field<arma::Cube<double>>& xs);
	/** Returns field of 2+deltas.size() gradient elements: dw, db, and rest are dxs */
	arma::field<arma::Cube<double>> getGrads(const arma::field<arma::Cube<double>>& deltas);
	arma::field<arma::Cube<double>> backProp(const arma::field<arma::Cube<double>>& deltas);
};

#endif /* LAYER_VANILLAFEEDFORWARD_HPP_ */
