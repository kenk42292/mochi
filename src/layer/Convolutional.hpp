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
	unsigned int mBatchSize;
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
	arma::field<arma::Cube<double>> rawmxs;
	arma::field<arma::Cube<double>> mws;
	arma::Cube<double> mbs;
	arma::field<arma::Cube<double>> mdwdb;
	double mWdecay;
	friend class ConvolutionalTest;
public:
	Convolutional(unsigned int batchSize,
			std::vector<unsigned int> inputDim,
			unsigned int numPatterns,
			std::vector<unsigned int> patternDim,
			std::vector<unsigned int> outputDim,
			Optimizer* optimizer, double wdecay);
	virtual ~Convolutional();

	/** Correlate all xs through a specific pattern */
	arma::Cube<double> feedForward(const arma::Mat<double>& x,
			const arma::Mat<double>& ws);
	arma::field<arma::Cube<double>> feedForward(
			const arma::field<arma::Cube<double>>& xs);
	arma::field<arma::Cube<double>> getGrads(const arma::field<arma::Cube<double>>& deltas);
	arma::field<arma::Cube<double>> backProp(
			const arma::field<arma::Cube<double>>& deltas);
	arma::Mat<double> im2col(const arma::Cube<double>& x, unsigned int h, unsigned int w, unsigned int d);
	arma::Mat<double> w2row(const arma::field<arma::Cube<double>>& w);
	arma::Mat<double> d2row(const arma::Cube<double>& delta);

};

#endif /* LAYER_CONVOLUTIONAL_HPP_ */
