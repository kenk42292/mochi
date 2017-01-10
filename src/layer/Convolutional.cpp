/*
 * Convolutional.cpp
 *
 *  Created on: Dec 23, 2016
 *      Author: ken
 */

#include "Convolutional.hpp"

Convolutional::Convolutional(unsigned int batchSize, std::vector<unsigned int> inputDim,
		unsigned int numPatterns, std::vector<unsigned int> patternDim,
		std::vector<unsigned int> outputDim, std::string mode, Optimizer* optimizer, double wdecay) :
		mBatchSize(batchSize),
		mInDepth(inputDim[0]), mInHeight(inputDim[1]), mInWidth(inputDim[2]),
		mNumPatterns(numPatterns),
		mPatternDepth(patternDim[0]), mPatternHeight(patternDim[1]), mPatternWidth(patternDim[2]),
		mOutDepth(outputDim[0]), mOutHeight(outputDim[1]), mOutWidth(outputDim[2]), mode(mode),
		mWdecay(wdecay) {
	mws = arma::field<arma::Cube<double>>(mNumPatterns);
	for (unsigned int i = 0; i < mNumPatterns; ++i) {
		mws[i] = arma::Cube<double>(mPatternHeight, mPatternWidth,
				mPatternDepth, arma::fill::randn);
	}
	mbs = arma::Cube<double>(1, 1, mNumPatterns, arma::fill::zeros);
	mdwdb = arma::field<arma::Cube<double>>(mNumPatterns + 1);
	mOptimizer = optimizer;
}

Convolutional::~Convolutional() {
	delete mOptimizer;
}

arma::Cube<double> Convolutional::feedForward(const arma::Mat<double>& x,
		const arma::Mat<double>& ws) {
	arma::Mat<double> yMat = (ws*x).t();
	arma::Cube<double> y(yMat.begin(), mOutHeight, mOutWidth, mOutDepth);
	for (unsigned int i=0; i<y.n_slices; ++i) {
		y.slice(i) += (double) mbs(0,0,i);
	}
	return y;
}

arma::field<arma::Cube<double>> Convolutional::feedForward(
		const arma::field<arma::Cube<double>>& xs) {
	arma::field<arma::Cube<double>> ys(xs.size());
	arma::Mat<double> ws = w2row(mws);
	mxs = arma::field<arma::Cube<double>>(xs.size());
	for (unsigned int i=0; i<xs.size(); ++i) {
		mxs[i] = arma::Cube<double>(xs[i].begin(), mInHeight, mInWidth, mInDepth);
	}
	for (unsigned int i = 0; i < mxs.size(); ++i) {
		arma::Mat<double> xMat = im2col(mxs[i], mPatternHeight, mPatternWidth, mPatternDepth);
		ys[i] = feedForward(xMat, ws);
	}
	return ys;
}

/**
 * Returns the gradients in a single field of cubes, in the order:
 * dw1, dw2, dw3...
 * db
 * dx1, dx2, dx3...
 */
arma::field<arma::Cube<double>> Convolutional::getGrads(
		const arma::field<arma::Cube<double>>& rawDeltas) {

	arma::field<arma::Cube<double>> deltas(rawDeltas.size());
	if (rawDeltas(0).n_slices != mOutDepth || rawDeltas(0).n_rows != mOutHeight
			|| rawDeltas(0).n_cols != mOutWidth) {
		for (unsigned int i = 0; i < rawDeltas.size(); ++i) {
			deltas(i) = arma::Cube<double>(rawDeltas(i).begin(), mOutHeight,
					mOutWidth, mOutDepth);
		}
	} else {
		deltas = rawDeltas;
	}

	arma::field<arma::Cube<double>> grads(mNumPatterns + 1 + deltas.size());
	for (unsigned int k = 0; k < mNumPatterns; ++k) {
		grads[k] = arma::Cube<double>(mPatternHeight, mPatternWidth,
				mPatternDepth, arma::fill::zeros);
	}
	grads(mNumPatterns) = arma::Cube<double>(1, 1, mNumPatterns,
			arma::fill::zeros);
	for (unsigned int i = 0; i < deltas.size(); ++i) {
		grads[i + mNumPatterns + 1] = arma::Cube<double>(mInHeight, mInWidth,
				mInDepth, arma::fill::zeros);
	}

	for (unsigned int i = 0; i < deltas.size(); ++i) { // Iterate through batch
		arma::Cube<double>& x = mxs[i];
		arma::Mat<double> xMat = im2col(mxs[i], deltas[i].n_rows, deltas[i].n_cols, 1);
		arma::Mat<double> deltaMat = d2row(deltas(i));
		for (unsigned int k = 0; k < mNumPatterns; ++k) { // Iterate through patterns
			arma::Mat<double> corrMat = deltaMat.row(k)*xMat;
			grads[k] += arma::Cube<double>(corrMat.begin(), mPatternHeight, mPatternWidth, mPatternDepth);
			for (unsigned int c = 0; c < x.n_slices; ++c) { // Iterate through x slices
				grads[i + mNumPatterns + 1].slice(c) += arma::conv2(
						deltas[i].slice(k), mws[k].slice(c), "full");
			}
		}
		grads[mNumPatterns] += arma::sum(arma::sum(deltas[i], 0), 1);
	}
	/*weight decay*/
	for (unsigned int k=0; k<mNumPatterns; ++k) {
		grads[k] += mWdecay*mws[k];
	}
	return grads;
}

arma::field<arma::Cube<double>> Convolutional::backProp(
		const arma::field<arma::Cube<double>>& deltas) {
	const arma::field<arma::Cube<double>>& grads = getGrads(deltas);
	const arma::field<arma::Cube<double>>& paramChanges = mOptimizer->delta(
			grads.rows(0, mNumPatterns), deltas.size());
	for (unsigned int i = 0; i < mws.size(); ++i) {
		mws[i] -= paramChanges[i];
	}

	mbs -= paramChanges[mNumPatterns];
	return grads.rows(mNumPatterns + 1, grads.size() - 1);
}

/** Converts a single input cube to an appropriate matrix.
 * The matrix's columns are vectorized subcubes of length h*w*d*/
arma::Mat<double> Convolutional::im2col(const arma::Cube<double>& x,
		unsigned int h, unsigned int w, unsigned int d) {

	unsigned int nHShifts = mInHeight - h + 1;
	unsigned int nWShifts = mInWidth - w + 1;
	unsigned int nDShifts = mInDepth - d + 1;

	arma::Mat<double> xMat(h * w * d, nHShifts*nWShifts*nDShifts);

	for (unsigned int k = 0; k < nDShifts; ++k) {
		for (unsigned int j = 0; j < nWShifts; ++j) {
			for (unsigned int i = 0; i < nHShifts; ++i) {
				xMat.col(k * nWShifts * nHShifts + j * nHShifts + i) =
						arma::vectorise(
								x.subcube(i, j, k, i + h - 1, j + w - 1,
										k + d - 1));
			}
		}
	}
	return xMat;
}

/** Converts ALL weight cubes into a single matrix - each w_k is a row in this matrix */
arma::Mat<double> Convolutional::w2row(
		const arma::field<arma::Cube<double>>& ws) {
	arma::Mat<double> wMat(mNumPatterns,
			mPatternHeight * mPatternWidth * mPatternDepth);
	for (unsigned int i = 0; i < mNumPatterns; ++i) {
		wMat.row(i) = arma::Mat<double>(ws(i).begin(), 1, ws(i).size());
	}
	return wMat;
}

/** Converts each slice of a delta cube into a row in a resultant matrix */
arma::Mat<double> Convolutional::d2row(const arma::Cube<double>& delta) {
	arma::Mat<double> dMat(mNumPatterns, mOutHeight * mOutWidth);
	for (unsigned int k = 0; k < mNumPatterns; ++k) {
		dMat.row(k) = arma::Mat<double>(delta.slice(k).begin(), 1, delta.n_cols*delta.n_rows);
	}
	return dMat;
}


