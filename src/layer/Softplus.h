/*
 * Softplus.h
 *
 *  Created on: Dec 11, 2016
 *      Author: ken
 */

#ifndef LAYER_SOFTPLUS_H_
#define LAYER_SOFTPLUS_H_

#include "Layer.h"

class Softplus: public Layer {
public:
	Softplus();
	virtual ~Softplus();

	arma::field<arma::Cube<double>> feedForward(const arma::field<arma::Cube<double>>& zs);
	arma::field<arma::Cube<double>> backProp(const arma::field<arma::Cube<double>>& deltas);
};

#endif /* LAYER_SOFTPLUS_H_ */