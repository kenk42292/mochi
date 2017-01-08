/*
 * Quadratic.h
 *
 *  Created on: Dec 11, 2016
 *      Author: ken
 */

#ifndef LOSS_QUADRATIC_HPP_
#define LOSS_QUADRATIC_HPP_

#include "Loss.hpp"

class Quadratic: public Loss {
public:
	Quadratic();
	virtual ~Quadratic();

	double loss(arma::Cube<double> output, arma::Cube<double> y);
	arma::Cube<double> loss_prime(const arma::Cube<double>& output, const arma::Cube<double>& y);

};

#endif /* LOSS_QUADRATIC_HPP_ */
