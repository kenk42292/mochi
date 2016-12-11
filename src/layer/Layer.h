/*
cd pa	 * Layer.h
 *
 *  Created on: Dec 8, 2016
 *      Author: ken
 */

#ifndef LAYER_LAYER_H_
#define LAYER_LAYER_H_

#include <armadillo>

class Layer {
public:
	Layer();
	virtual ~Layer();

	/* Batch processing */
	virtual std::vector<arma::Col<double>> feedForward(const std::vector<arma::Col<double>>& zs);
	virtual std::vector<arma::Mat<double>> feedForward(const std::vector<arma::Mat<double>>& zs);
	virtual std::vector<arma::Cube<double>> feedForward(const std::vector<arma::Cube<double>>& zs);
	virtual std::vector<arma::Col<double>> backProp(const std::vector<arma::Col<double>>& deltas);
	virtual std::vector<arma::Mat<double>> backProp(const std::vector<arma::Mat<double>>& deltas);
	virtual std::vector<arma::Cube<double>> backProp(const std::vector<arma::Cube<double>>& deltas);
};

#endif /* LAYER_LAYER_H_ */
