/*
 * Loss.h
 *
 *  Created on: Dec 11, 2016
 *      Author: ken
 */

#ifndef LOSS_H_
#define LOSS_H_

#include <armadillo>

class Loss {
public:
	Loss();
	virtual ~Loss();

	virtual double loss(arma::Cube<double> output, arma::Cube<double> y);
	virtual double loss(arma::field<arma::Cube<double>> outputs, arma::field<arma::Cube<double>> ys);
	virtual arma::Cube<double> loss_prime(const arma::Cube<double>& output, const arma::Cube<double>& y);
	virtual arma::field<arma::Cube<double>> loss_prime(const arma::field<arma::Cube<double>>& outputs, const arma::field<arma::Cube<double>>& ys);
};

#endif /* LOSS_H_ */
