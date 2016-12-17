/*
 * Utils.h
 *
 *  Created on: Dec 16, 2016
 *      Author: ken
 */

#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <cstdlib>
#include <armadillo>

class Utils {
public:
	Utils();
	virtual ~Utils();

	static void shuffle(arma::field<arma::Cube<double>>& inputs, arma::field<arma::Cube<double>>& outputs);
};

#endif /* UTILS_HPP_ */
