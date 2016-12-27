/*
 * Convolutional.hpp
 *
 *  Created on: Dec 23, 2016
 *      Author: ken
 */

#ifndef LAYER_CONVOLUTIONAL_HPP_
#define LAYER_CONVOLUTIONAL_HPP_

#include "Layer.hpp"
#include "../Utils.hpp"

class Convolutional: public Layer {
private:
	// TODO: This code is ugly. At least make constructor args const vect. refs
	unsigned int mInDepth;
	unsigned int mInHeight;
	unsigned int mInWidth;
	unsigned int mNumPatterns;
	unsigned int mPatternDepth;
	unsigned int mPatternHeight;
	unsigned int mPatternWidth;
	unsigned int mOutDepth;
	unsigned int mOutHeight;
	unsigned int mOutWidth;
	Optimizer* mOptimizer;
	arma::field<arma::Cube<double>> mxs;
	arma::field<arma::Cube<double>> mws;
	arma::field<arma::Cube<double>> mbs;
public:
	Convolutional(std::vector<unsigned int> inputDim,
			unsigned int numPatterns,
			std::vector<unsigned int> patternDim,
			std::vector<unsigned int> outputDim,
			Optimizer* optimizer);
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
