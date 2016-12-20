/*
 * RMSProp.hpp
 *
 *  Created on: Dec 19, 2016
 *      Author: ken
 */

#ifndef LAYER_OPTIMIZER_RMSPROP_HPP_
#define LAYER_OPTIMIZER_RMSPROP_HPP_

#include "Optimizer.hpp"

class RMSProp: public Optimizer {
private:
	double mEta;

public:
	RMSProp(double eta);
	virtual ~RMSProp();
	arma::field<arma::Cube<double>> delta(const arma::field<arma::Cube<double>>& gradients, unsigned int batchSize);
};

#endif /* LAYER_OPTIMIZER_RMSPROP_HPP_ */
