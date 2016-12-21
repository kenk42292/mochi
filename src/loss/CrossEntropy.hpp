/*
 * CrossEntropy.h
 *
 *  Created on: Dec 11, 2016
 *      Author: ken
 */

#ifndef LOSS_CROSSENTROPY_HPP_
#define LOSS_CROSSENTROPY_HPP_

#include "Loss.hpp"

class CrossEntropy: public Loss {
public:
	CrossEntropy();
	virtual ~CrossEntropy();

	double loss(arma::Cube<double> output, arma::Cube<double> y);
	double loss(arma::field<arma::Cube<double>> outputs, arma::field<arma::Cube<double>> ys);
	arma::Cube<double> loss_prime(const arma::Cube<double>& output, const arma::Cube<double>& y);
	arma::field<arma::Cube<double>> loss_prime(const arma::field<arma::Cube<double>>& outputs, const arma::field<arma::Cube<double>>& ys);
};

#endif /* LOSS_CROSSENTROPY_HPP_ */
