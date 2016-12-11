/*
 * Sigmoid.h
 *
 *  Created on: Dec 8, 2016
 *      Author: ken
 */

#ifndef LAYER_SIGMOID_H_
#define LAYER_SIGMOID_H_

#include "Layer.h"
#include <armadillo>

class Sigmoid: public Layer {
public:
	Sigmoid();
	virtual ~Sigmoid();

	arma::field<arma::Cube<double>> feedForward(const arma::field<arma::Cube<double>>& zs);
	arma::field<arma::Cube<double>> backProp(const arma::field<arma::Cube<double>>& deltas);
};

#endif /* LAYER_SIGMOID_H_ */
