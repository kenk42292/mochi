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
private:
	arma::Mat<double> mW;
	arma::Mat<double> mB;
public:
	VanillaFeedForward();
	virtual ~VanillaFeedForward();

	arma::Cube<double> VanillaFeedForward::feedForward(const arma::Cube<double>& z);
	arma::field<arma::Cube<double>> feedForward(const arma::field<arma::Cube<double>>& zs);
	arma::field<arma::Cube<double>> backProp(const arma::field<arma::Cube<double>>& deltas);
};

#endif /* LAYER_VANILLAFEEDFORWARD_H_ */
