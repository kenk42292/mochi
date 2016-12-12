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
private:

public:
	Layer();
	virtual ~Layer();

	virtual arma::field<arma::Cube<double>> feedForward(const arma::field<arma::Cube<double>>& zs);
	virtual arma::field<arma::Cube<double>> backProp(const arma::field<arma::Cube<double>>& deltas);
};

#endif /* LAYER_LAYER_H_ */
