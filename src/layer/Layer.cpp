/*
 * Layer.cpp
 *
 *  Created on: Dec 8, 2016
 *      Author: ken
 */

#include "Layer.h"

Layer::Layer() {
	// TODO Auto-generated constructor stub

}

Layer::~Layer() {
	// TODO Auto-generated destructor stub
}

arma::Mat<double> Layer::feedForward(arma::Mat<double> z) {
	return z;
}

arma::Mat<double> Layer::backProp(arma::Mat<double> delta) {
	return delta;
}

