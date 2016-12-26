/*
 * Convolutional.cpp
 *
 *  Created on: Dec 23, 2016
 *      Author: ken
 */

#include "Convolutional.hpp"

Convolutional::Convolutional(
		unsigned int numPatterns,
		unsigned int patternDepth,
		unsigned int patternHeight,
		unsigned int patternWidth) :
			mNumPatterns(numPatterns),
			mPatternDepth(patternDepth),
			mPatternHeight(patternHeight),
			mPatternWidth(patternWidth) {
	mWs = arma::field<arma::Cube<double>>(mNumPatterns);
	mBs = arma::field<arma::Cube<double>>(mNumPatterns);
	for (unsigned int i = 0; i < mNumPatterns; ++i) {
		mWs[i] = arma::Cube<double>(mPatternHeight, mPatternWidth, mPatternDepth,
				arma::fill::randn);
		mWs[i] = arma::Cube<double>(mPatternHeight, mPatternWidth, mPatternDepth,
				arma::fill::zeros);
	}
}

Convolutional::~Convolutional() {}

arma::Cube<double> Convolutional::feedForward(const arma::Cube<double>& x,
		const arma::field<arma::Cube<double>>& flippedWeights) {
	arma::Cube<double> y(x.n_rows - mPatternHeight + 1,	x.n_cols - mPatternWidth + 1, mNumPatterns);
	for (unsigned int i = 0; i < mNumPatterns; ++i) {
		arma::Mat<double> ySlice(x.n_rows + mPatternHeight - 1, x.n_cols + mPatternWidth - 1,
				arma::fill::zeros);
		for (unsigned int j=0; j<x.n_slices; ++j) {
			ySlice += arma::conv2(x.slice(j), flippedWeights[i].slice(j), "full");
		}
		y.slice(i) = ySlice.submat(mPatternHeight-1, mPatternWidth-1, x.n_rows-1, x.n_cols-1);
	}
	return y;
}

arma::field<arma::Cube<double>> Convolutional::feedForward(
		const arma::field<arma::Cube<double>>& xs) {
	arma::field<arma::Cube<double>> ys(xs.size());
	arma::field<arma::Cube<double>> flippedWeights(mNumPatterns);
	/** For cross-correlation, must use flipped weights and convolution. */
	for (unsigned int i = 0; i < mNumPatterns; ++i) {
		for (unsigned int j = 0; j < mWs[i].n_slices; ++j) {
			flippedWeights[i].slice(j) = arma::fliplr(
					arma::flipud(mWs[i].slice(j)));
		}
	}
	for (unsigned int i = 0; i < xs.size(); ++i) {
		ys[i] = feedForward(xs[i], flippedWeights);
	}
	return ys;
}

arma::field<arma::Cube<double>> Convolutional::backProp(
		const arma::field<arma::Cube<double>>& deltas) {

}

