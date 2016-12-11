/*
 * Sigmoid.h
 *
 *  Created on: Dec 8, 2016
 *      Author: ken
 */

#ifndef LAYER_SIGMOID_H_
#define LAYER_SIGMOID_H_

#include <armadillo>

class Sigmoid {
public:
	Sigmoid();
	virtual ~Sigmoid();

	/* Single samples */
	arma::Col<double> feedForward(const arma::Col<double>& z);
	arma::Mat<double> feedForward(const arma::Mat<double>& z);
	arma::Cube<double> feedForward(const arma::Cube<double>& z);
	arma::Col<double> backProp(const arma::Col<double>& delta);
	arma::Mat<double> backProp(const arma::Mat<double>& delta);
	arma::Cube<double> backProp(const arma::Cube<double>& delta);

	/* Batch processing */
	std::vector<arma::Col<double>> feedForward(const std::vector<arma::Col<double>>& zs);
	std::vector<arma::Mat<double>> feedForward(const std::vector<arma::Mat<double>>& zs);
	std::vector<arma::Cube<double>> feedForward(const std::vector<arma::Cube<double>>& zs);
	std::vector<arma::Col<double>> backProp(const std::vector<arma::Col<double>>& deltas);
	std::vector<arma::Mat<double>> backProp(const std::vector<arma::Mat<double>>& deltas);
	std::vector<arma::Cube<double>> backProp(const std::vector<arma::Cube<double>>& deltas);
};

#endif /* LAYER_SIGMOID_H_ */
