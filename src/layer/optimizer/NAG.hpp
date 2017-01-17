/*
 * NAG.hpp
 *
 *  Created on: Jan 17, 2017
 *      Author: ken
 */

#ifndef LAYER_OPTIMIZER_NAG_HPP_
#define LAYER_OPTIMIZER_NAG_HPP_

#include "Optimizer.hpp"

class NAG: public Optimizer {
private:
	double mEta;
	bool cacheInitialized;
	arma::field<arma::Cube<double>> mMomentum;
	double mGamma;
public:
	NAG(double eta, double gamma);
	virtual ~NAG();
	arma::field<arma::Cube<double>> delta(
			const arma::field<arma::Cube<double>>& gradients,
			unsigned int batchSize);
};

#endif /* LAYER_OPTIMIZER_NAG_HPP_ */
