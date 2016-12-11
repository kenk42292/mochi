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
	arma::Col<double> feedForward(arma::Col<double> z);
	arma::Mat<double> feedForward(arma::Mat<double> z);
	arma::Cube<double> feedForward(arma::Cube<double> z);
	arma::Col<double> backProp(arma::Col<double> delta);
	arma::Mat<double> backProp(arma::Mat<double> delta);
	arma::Cube<double> backProp(arma::Cube<double> delta);
};

#endif /* LAYER_SIGMOID_H_ */
