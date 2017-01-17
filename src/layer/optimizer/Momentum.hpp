/*
 * Momentum.hpp
 *
 *  Created on: Jan 16, 2017
 *      Author: ken
 */

#ifndef LAYER_OPTIMIZER_MOMENTUM_HPP_
#define LAYER_OPTIMIZER_MOMENTUM_HPP_

#include "Optimizer.hpp"

class Momentum: public Optimizer {
private:
	double mEta;
	bool cacheInitialized;
	arma::field<arma::Cube<double>> mCache;
	double mGamma;
	double eps;
public:
	Momentum(double eta, double gamma);
	virtual ~Momentum();
	arma::field<arma::Cube<double>> delta(
			const arma::field<arma::Cube<double>>& gradients,
			unsigned int batchSize);
};

#endif /* LAYER_OPTIMIZER_MOMENTUM_HPP_ */
