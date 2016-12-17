/*
 * Softmax.h
 *
 *  Created on: Dec 11, 2016
 *      Author: ken
 */

#ifndef LAYER_SOFTMAX_HPP_
#define LAYER_SOFTMAX_HPP_

#include "Layer.hpp"

class Softmax: public Layer {
private:
	arma::field<arma::Cube<double>> mYs;
public:
	Softmax();
	virtual ~Softmax();

	arma::Cube<double> softmax(const arma::Cube<double>& z);
	arma::field<arma::Cube<double>> feedForward(const arma::field<arma::Cube<double>>& zs);
	arma::field<arma::Cube<double>> backProp(const arma::field<arma::Cube<double>>& deltas);
};

#endif /* LAYER_SOFTMAX_HPP_ */
