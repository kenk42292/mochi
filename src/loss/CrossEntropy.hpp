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
	arma::Cube<double> loss_prime(const arma::Cube<double>& output, const arma::Cube<double>& y);
};

#endif /* LOSS_CROSSENTROPY_HPP_ */
