/*
 * Optimizer.h
 *
 *  Created on: Dec 19, 2016
 *      Author: ken
 */

#ifndef LAYER_OPTIMIZER_OPTIMIZER_HPP_
#define LAYER_OPTIMIZER_OPTIMIZER_HPP_

#include <armadillo>

class Optimizer {
public:
	Optimizer();
	virtual ~Optimizer();
	virtual arma::field<arma::Cube<double>> delta(const arma::field<arma::Cube<double>>& gradients, unsigned int batchSize);
};

#endif /* LAYER_OPTIMIZER_OPTIMIZER_HPP_ */
