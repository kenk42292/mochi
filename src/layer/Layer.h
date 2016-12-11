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

	/* Single samples */
	virtual arma::Col<double> feedForward(const arma::Col<double>& z);
	virtual arma::Mat<double> feedForward(const arma::Mat<double>& z);
	virtual arma::Cube<double> feedForward(const arma::Cube<double>& z);
	virtual arma::Col<double> backProp(const arma::Col<double>& delta);
	virtual arma::Mat<double> backProp(const arma::Mat<double>& delta);
	virtual arma::Cube<double> backProp(const arma::Cube<double>& delta);

	/* Batch processing */
	virtual std::vector<arma::Col<double>> feedForward(const std::vector<arma::Col<double>>& zs);
	virtual std::vector<arma::Mat<double>> feedForward(const std::vector<arma::Mat<double>>& zs);
	virtual std::vector<arma::Cube<double>> feedForward(const std::vector<arma::Cube<double>>& zs);
	virtual std::vector<arma::Col<double>> backProp(const std::vector<arma::Col<double>>& deltas);
	virtual std::vector<arma::Mat<double>> backProp(const std::vector<arma::Mat<double>>& deltas);
	virtual std::vector<arma::Cube<double>> backProp(const std::vector<arma::Cube<double>>& deltas);
};

#endif /* LAYER_LAYER_H_ */
