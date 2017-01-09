/*
 * Loss.h
 *
 *  Created on: Dec 11, 2016
 *      Author: ken
 */

#ifndef LOSS_H_
#define LOSS_H_

#include <armadillo>
#include "../layer/Layer.hpp"

/**
 * While the accumulating methods, LOSS and LOSS_PRIME for fields should stay fixed, over-ride
 * the single-sample methods of the same name in the children class.
 * */
class Loss {
public:
	Loss();
	virtual ~Loss();

	virtual double loss(arma::Cube<double> output, arma::Cube<double> y);
	double loss(arma::field<arma::Cube<double>> outputs, arma::field<arma::Cube<double>> ys);

	virtual arma::Cube<double> loss_prime(const arma::Cube<double>& output, const arma::Cube<double>& y);
	arma::field<arma::Cube<double>> loss_prime(const arma::field<arma::Cube<double>>& outputs, const arma::field<arma::Cube<double>>& ys);
};

#endif /* LOSS_H_ */
