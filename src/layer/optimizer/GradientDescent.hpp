/*
 * GradientDescent.h
 *
 *  Created on: Dec 19, 2016
 *      Author: ken
 */

#ifndef LAYER_OPTIMIZER_GRADIENTDESCENT_HPP_
#define LAYER_OPTIMIZER_GRADIENTDESCENT_HPP_

#include "Optimizer.hpp"
#include <armadillo>

class GradientDescent: public Optimizer {
private:
	double mEta;
public:
	GradientDescent(double eta);
	virtual ~GradientDescent();
	arma::field<arma::Cube<double>> delta(const arma::field<arma::Cube<double>>& gradients, unsigned int batchSize);
};

#endif /* LAYER_OPTIMIZER_GRADIENTDESCENT_HPP_ */
