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
	double mGamma;
	bool cacheInitialized;
	arma::field<arma::Cube<double>> mCache;
	double eps;

public:
	RMSProp(double eta, double gamma);
	virtual ~RMSProp();
	arma::field<arma::Cube<double>> delta(const arma::field<arma::Cube<double>>& gradients, unsigned int batchSize);
};

#endif /* LAYER_OPTIMIZER_RMSPROP_HPP_ */
