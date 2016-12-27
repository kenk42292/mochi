/*
 * Convolutional.cpp
 *
 *  Created on: Dec 23, 2016
 *      Author: ken
 */

#include "Convolutional.hpp"

Convolutional::Convolutional(std::vector<unsigned int> inputDim,
		unsigned int numPatterns, std::vector<unsigned int> patternDim,
		std::vector<unsigned int> outputDim, Optimizer* optimizer) :
		mNumPatterns(numPatterns), mInDepth(inputDim[0]), mInHeight(
				inputDim[1]), mInWidth(inputDim[2]), mOutDepth(outputDim[0]), mOutHeight(
				outputDim[1]), mOutWidth(outputDim[2]), mPatternDepth(
				patternDim[0]), mPatternHeight(patternDim[1]), mPatternWidth(
				patternDim[2]) {
	mws = arma::field<arma::Cube<double>>(mNumPatterns);
	mbs = arma::field<arma::Cube<double>>(mNumPatterns);
	mOptimizer = optimizer;
	for (unsigned int i = 0; i < mNumPatterns; ++i) {
		mws[i] = arma::Cube<double>(mPatternHeight, mPatternWidth,
				mPatternDepth, arma::fill::randn);
		mbs[i] = arma::Cube<double>(mPatternHeight, mPatternWidth,
				mPatternDepth, arma::fill::zeros);
	}
}

Convolutional::~Convolutional() {
	delete mOptimizer;
}

arma::Cube<double> Convolutional::feedForward(const arma::Cube<double>& x,
		const arma::field<arma::Cube<double>>& flippedWeights) {
	arma::Cube<double> y(x.n_rows - mPatternHeight + 1,
			x.n_cols - mPatternWidth + 1, mNumPatterns);
	for (unsigned int i = 0; i < mNumPatterns; ++i) {
		arma::Mat<double> ySlice(x.n_rows + mPatternHeight - 1,
				x.n_cols + mPatternWidth - 1, arma::fill::zeros);
		for (unsigned int j = 0; j < x.n_slices; ++j) {
			ySlice += arma::conv2(x.slice(j), flippedWeights[i].slice(j),
					"full");
		}
		y.slice(i) = ySlice.submat(mPatternHeight - 1, mPatternWidth - 1,
				x.n_rows - 1, x.n_cols - 1);
	}
	// TODO: Add the bias term
	return y;
}

arma::field<arma::Cube<double>> Convolutional::feedForward(
		const arma::field<arma::Cube<double>>& xs) {

	mxs = xs;
	arma::field<arma::Cube<double>> ys(xs.size());

	/** For cross-correlation, must use flipped weights and convolution. */
	arma::field<arma::Cube<double>> flippedWeights = Utils::flipCubes(mws);

	for (unsigned int i = 0; i < xs.size(); ++i) {
		ys[i] = feedForward(xs[i], flippedWeights);
	}
	return ys;
}

arma::field<arma::Cube<double>> Convolutional::backProp(
		const arma::field<arma::Cube<double>>& deltas) {


	arma::field<arma::Cube<double>> dws(mNumPatterns);
	arma::field<arma::Cube<double>> dxs(deltas.size());
	arma::field<arma::Cube<double>> dbs(mNumPatterns);

	/** Flipped deltas for cross-correlation */
	arma::field<arma::Cube<double>> flippedDeltas = Utils::flipCubes(deltas);

	for (unsigned int k = 0; k < mNumPatterns; ++k) {
		dws[k] = arma::Cube<double>(mPatternHeight, mPatternWidth,
				mPatternDepth, arma::fill::zeros);
		for (unsigned int i = 0; i < deltas.size(); ++i) {
			for (unsigned int c = 0; c < mxs[i].n_slices; ++c) {
				const arma::Mat<double>& fullConv = arma::conv2(mxs[i].slice(c),
						flippedDeltas[i].slice(c), "full");
				dws[k].slice(c) += fullConv.submat(flippedDeltas[i].n_rows - 1,
						flippedDeltas[i].n_cols - 1, mxs[i].n_rows - 1,
						mxs[i].n_cols - 1);
			}
		}
	}
	arma::field<arma::Cube<double>> paramChanges = mOptimizer->delta(dws,
			deltas.size());
	for (unsigned int i = 0; i < mws.size(); ++i) {
		mws[i] -= paramChanges[i];
	}
	return arma::field<arma::Cube<double>>(1); //TODO: Complete with db, dx
}

