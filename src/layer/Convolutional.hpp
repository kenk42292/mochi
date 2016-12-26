/*
 * Convolutional.hpp
 *
 *  Created on: Dec 23, 2016
 *      Author: ken
 */

#ifndef LAYER_CONVOLUTIONAL_HPP_
#define LAYER_CONVOLUTIONAL_HPP_

#include "Layer.hpp"

class Convolutional: public Layer {
private:
	unsigned int mNumPatterns;
	unsigned int mPatternDepth;
	unsigned int mPatternHeight;
	unsigned int mPatternWidth;
	arma::field<arma::Cube<double>> mWs;
	arma::field<arma::Cube<double>> mBs;
public:
	Convolutional(unsigned int numPatterns, unsigned int patternDepth,
			unsigned int patternHeight, unsigned int patternWidth);
	virtual ~Convolutional();

	/** Correlate all xs through a specific pattern */
	arma::Cube<double> feedForward(const arma::Cube<double>& x,
			const arma::field<arma::Cube<double>>& flippedWeights);
	virtual arma::field<arma::Cube<double>> feedForward(
			const arma::field<arma::Cube<double>>& xs);
	virtual arma::field<arma::Cube<double>> backProp(
			const arma::field<arma::Cube<double>>& deltas);
};

#endif /* LAYER_CONVOLUTIONAL_HPP_ */
