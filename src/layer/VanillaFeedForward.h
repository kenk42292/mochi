/*
 * VanillaFeedForward.h
 *
 *  Created on: Dec 11, 2016
 *      Author: ken
 */

#ifndef LAYER_VANILLAFEEDFORWARD_H_
#define LAYER_VANILLAFEEDFORWARD_H_

#include "Layer.h"

class VanillaFeedForward: public Layer {
public:
	arma::Mat<double> mW;
	arma::Col<double> mB;
	/** Stored batch inputs for back propagation */
	arma::field<arma::Cube<double>> mxs;
public:
	VanillaFeedForward(unsigned int nIn, unsigned int nOut);
	virtual ~VanillaFeedForward();

	arma::Cube<double> feedForward(const arma::Cube<double>& x);
	arma::field<arma::Cube<double>> feedForward(const arma::field<arma::Cube<double>>& xs);
	arma::field<arma::Cube<double>> backProp(const arma::field<arma::Cube<double>>& deltas);
};

#endif /* LAYER_VANILLAFEEDFORWARD_H_ */
