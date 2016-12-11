/*
 * Softmax.h
 *
 *  Created on: Dec 11, 2016
 *      Author: ken
 */

#ifndef LAYER_SOFTMAX_H_
#define LAYER_SOFTMAX_H_

#include "Layer.h"

class Softmax: public Layer {
public:
	Softmax();
	virtual ~Softmax();

	/* Batch processing */
	virtual std::vector<arma::Col<double>> feedForward(const std::vector<arma::Col<double>>& zs);
	virtual std::vector<arma::Mat<double>> feedForward(const std::vector<arma::Mat<double>>& zs);
	virtual std::vector<arma::Cube<double>> feedForward(const std::vector<arma::Cube<double>>& zs);
	virtual std::vector<arma::Col<double>> backProp(const std::vector<arma::Col<double>>& deltas);
	virtual std::vector<arma::Mat<double>> backProp(const std::vector<arma::Mat<double>>& deltas);
	virtual std::vector<arma::Cube<double>> backProp(const std::vector<arma::Cube<double>>& deltas);
};

#endif /* LAYER_SOFTMAX_H_ */
