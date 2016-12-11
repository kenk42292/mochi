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
public:
	Layer();
	virtual ~Layer();
	virtual arma::Mat<double> feedForward(arma::Mat<double> vec);
	virtual arma::Mat<double> backProp(arma::Mat<double> vec);
};

#endif /* LAYER_LAYER_H_ */
