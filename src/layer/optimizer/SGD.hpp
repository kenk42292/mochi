/*
 * GradientDescent.h
 *
 *  Created on: Dec 19, 2016
 *      Author: ken
 */

#ifndef LAYER_OPTIMIZER_SGD_HPP_
#define LAYER_OPTIMIZER_SGD_HPP_

#include "Optimizer.hpp"
#include <armadillo>

class SGD: public Optimizer {
private:
	double mEta;
public:
	SGD(double eta);
	virtual ~SGD();
	arma::field<arma::Cube<double>> delta(const arma::field<arma::Cube<double>>& gradients, unsigned int batchSize);
};

#endif /* LAYER_OPTIMIZER_SGD_HPP_ */
