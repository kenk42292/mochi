/*
 * Adagrad.hpp
 *
 *  Created on: Dec 19, 2016
 *      Author: ken
 */

#ifndef LAYER_OPTIMIZER_ADAGRAD_HPP_
#define LAYER_OPTIMIZER_ADAGRAD_HPP_

#include "Optimizer.hpp"

class Adagrad: public Optimizer {
private:
	double mEta;
	bool cacheInitialized;
	arma::field<arma::Cube<double>> mCache;
	double eps;
public:
	Adagrad(double eta);
	virtual ~Adagrad();
	arma::field<arma::Cube<double>> delta(const arma::field<arma::Cube<double>>& gradients, unsigned int batchSize);
};

#endif /* LAYER_OPTIMIZER_ADAGRAD_HPP_ */
