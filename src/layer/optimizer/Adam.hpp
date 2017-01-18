/*
 * Adam.hpp
 *
 *  Created on: Jan 17, 2017
 *      Author: ken
 */

#ifndef LAYER_OPTIMIZER_ADAM_HPP_
#define LAYER_OPTIMIZER_ADAM_HPP_

#include "Optimizer.hpp"
#include <cmath>

/** Adam optimizer as described in this paper: https://arxiv.org/pdf/1412.6980v8.pdf */
class Adam: public Optimizer {
private:
	double mEta;
	bool cacheInitialized;
	arma::field<arma::Cube<double>> mM;
	arma::field<arma::Cube<double>> mV;
	double b1, b2;
	double t = 0;
	double eps = 1e-8;
public:
	Adam(double eta, double b1, double b2);
	virtual ~Adam();
	arma::field<arma::Cube<double>> delta(
				const arma::field<arma::Cube<double>>& gradients,
				unsigned int batchSize);
};

#endif /* LAYER_OPTIMIZER_ADAM_HPP_ */
