/*
cd pa	 * Layer.h
 *
 *  Created on: Dec 8, 2016
 *      Author: ken
 */

#ifndef LAYER_LAYER_HPP_
#define LAYER_LAYER_HPP_

#include <armadillo>
#include "optimizer/Optimizer.hpp"

class Layer {
private:

public:
	Layer();
	virtual ~Layer();

	virtual arma::field<arma::Cube<double>> feedForward(const arma::field<arma::Cube<double>>& xs);
	virtual arma::field<arma::Cube<double>> backProp(const arma::field<arma::Cube<double>>& deltas);
};

#endif /* LAYER_LAYER_HPP_ */
