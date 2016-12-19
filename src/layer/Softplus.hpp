/*
 * Softplus.h
 *
 *  Created on: Dec 11, 2016
 *      Author: ken
 */

#ifndef LAYER_SOFTPLUS_HPP_
#define LAYER_SOFTPLUS_HPP_

#include "Layer.hpp"

class Softplus: public Layer {
private:
	/* Needed for back-propagation */
	arma::field<arma::Cube<double>> mxs;
public:
	Softplus();
	virtual ~Softplus();

	arma::field<arma::Cube<double>> feedForward(const arma::field<arma::Cube<double>>& xs);
	arma::field<arma::Cube<double>> backProp(const arma::field<arma::Cube<double>>& deltas);
};

#endif /* LAYER_SOFTPLUS_HPP_ */
