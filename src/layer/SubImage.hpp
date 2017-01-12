/*
 * SubImage.hpp
 *
 *  Created on: Jan 11, 2017
 *      Author: ken
 */

#ifndef LAYER_SUBIMAGE_HPP_
#define LAYER_SUBIMAGE_HPP_

#include "Layer.hpp"
#include <armadillo>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

class SubImage: public Layer {
private:
	unsigned int mInHeight;
	unsigned int mInWidth;
	unsigned int mOutHeight;
	unsigned int mOutWidth;
public:
	SubImage(unsigned int inHeight, unsigned int inWidth, unsigned int outHeight, unsigned int outWidth);
	virtual ~SubImage();

	arma::field<arma::Cube<double>> feedForward(const arma::field<arma::Cube<double>>& xs);
	arma::field<arma::Cube<double>> backProp(const arma::field<arma::Cube<double>>& deltas);
};

#endif /* LAYER_SUBIMAGE_HPP_ */
