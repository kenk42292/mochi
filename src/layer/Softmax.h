/*
 * Softmax.h
 *
 *  Created on: Dec 11, 2016
 *      Author: ken
 */

#ifndef LAYER_SOFTMAX_H_
#define LAYER_SOFTMAX_H_

#include "Layer.h"

class Softmax: public Layer {
public:
	Softmax();
	virtual ~Softmax();

	arma::field<arma::Cube<double>> feedForward(const arma::field<arma::Cube<double>>& zs);
	arma::field<arma::Cube<double>> backProp(const arma::field<arma::Cube<double>>& deltas);
};

#endif /* LAYER_SOFTMAX_H_ */
