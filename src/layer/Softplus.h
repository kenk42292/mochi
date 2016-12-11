/*
 * Softplus.h
 *
 *  Created on: Dec 11, 2016
 *      Author: ken
 */

#ifndef LAYER_SOFTPLUS_H_
#define LAYER_SOFTPLUS_H_

#include "Layer.h"

class Softplus: public Layer {
public:
	Softplus();
	virtual ~Softplus();

	virtual std::vector<arma::Col<double>> feedForward(const std::vector<arma::Col<double>>& zs);
	virtual std::vector<arma::Mat<double>> feedForward(const std::vector<arma::Mat<double>>& zs);
	virtual std::vector<arma::Cube<double>> feedForward(const std::vector<arma::Cube<double>>& zs);
	virtual std::vector<arma::Col<double>> backProp(const std::vector<arma::Col<double>>& deltas);
	virtual std::vector<arma::Mat<double>> backProp(const std::vector<arma::Mat<double>>& deltas);
	virtual std::vector<arma::Cube<double>> backProp(const std::vector<arma::Cube<double>>& deltas);
};

#endif /* LAYER_SOFTPLUS_H_ */
