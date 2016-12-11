/*
 * Quadratic.h
 *
 *  Created on: Dec 11, 2016
 *      Author: ken
 */

#ifndef LOSS_QUADRATIC_H_
#define LOSS_QUADRATIC_H_

#include "Loss.h"

class Quadratic: public Loss {
public:
	Quadratic();
	virtual ~Quadratic();

	/** Single Sample Loss*/
	double loss(arma::Col<double> output, arma::Col<double> y);
	arma::Col<double> loss_prime(arma::Col<double> output, arma::Col<double> y);

};

#endif /* LOSS_QUADRATIC_H_ */
