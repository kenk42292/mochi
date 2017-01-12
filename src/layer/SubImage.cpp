/*
 * SubImage.cpp
 *
 *  Created on: Jan 11, 2017
 *      Author: ken
 */

#include "SubImage.hpp"

SubImage::SubImage(unsigned int inHeight, unsigned int inWidth, unsigned int outHeight, unsigned int outWidth) :
	mInHeight(inHeight), mInWidth(inWidth), mOutHeight(outHeight), mOutWidth(outWidth) {
	srand(time(NULL));
}

SubImage::~SubImage() {}

arma::field<arma::Cube<double>> SubImage::feedForward(const arma::field<arma::Cube<double>>& xs) {
	arma::field<arma::Cube<double>> ys(xs.size());
	for (unsigned int i=0; i<xs.size(); ++i) {
		unsigned int y = rand()%(mInHeight-mOutHeight);
		unsigned int x = rand()%(mInWidth-mOutWidth);
		ys(i) = xs(i).tube(y, x, y+mOutHeight, x+mOutWidth);
	}
	return ys;
}

arma::field<arma::Cube<double>> SubImage::backProp(const arma::field<arma::Cube<double>>& deltas) {
	//Shouldn't need for now...
	return arma::field<arma::Cube<double>>(0);
}


