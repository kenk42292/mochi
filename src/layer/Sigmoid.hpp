/*
 * Sigmoid.h
 *
 *  Created on: Dec 8, 2016
 *      Author: ken
 */

#ifndef LAYER_SIGMOID_HPP_
#define LAYER_SIGMOID_HPP_

#include <armadillo>
#include "Layer.hpp"

class Sigmoid: public Layer {
public:
	Sigmoid();
	virtual ~Sigmoid();

	arma::field<arma::Cube<double>> feedForward(const arma::field<arma::Cube<double>>& zs);
	arma::field<arma::Cube<double>> backProp(const arma::field<arma::Cube<double>>& deltas);
};

#endif /* LAYER_SIGMOID_HPP_ */