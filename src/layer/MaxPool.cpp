/*
 * MaxPool.cpp
 *
 *  Created on: Jan 2, 2017
 *      Author: ken
 */

#include "MaxPool.hpp"

MaxPool::MaxPool(std::vector<unsigned int> inputDim,
		std::vector<unsigned int> fieldDim,
		std::vector<unsigned int> outputDim) {
	mInDepth = inputDim[0];
	mInHeight = inputDim[1];
	mInWidth = inputDim[2];
	mFieldHeight = fieldDim[0];
	mFieldWidth = fieldDim[1];
	mOutDepth = outputDim[0];
	mOutHeight = outputDim[1];
	mOutWidth = outputDim[2];
}

MaxPool::~MaxPool() {}


arma::Cube<double> MaxPool::feedForward(const arma::Cube<double>& x,
		unsigned int sampleNum) {
	arma::Col<double> yFlattened(mInDepth*mInHeight*mInWidth/mFieldHeight/mFieldWidth);
	arma::Col<unsigned int> mIndices(mInDepth*mInHeight*mInWidth/mFieldHeight/mFieldWidth);
	double m;
	unsigned int mIndex;
	unsigned int depthOffset = 0;
	unsigned int resultIndex = 0;

	for (unsigned int k=0; k<mInDepth; ++k) {
		depthOffset = k*mInHeight*mInWidth;
		for (unsigned int j=0; j<mInWidth; j+=mFieldWidth) {
			for (unsigned int i=0; i<mInHeight; i+=mFieldHeight) {
				mIndex =depthOffset+j*mInHeight+i;
				m = x[mIndex];
				for (unsigned int q=j; q<j+mFieldWidth; ++q) { //top-left corner of field
					for (unsigned int p=i; p<i+mFieldHeight; ++p) {
						if (x[depthOffset+q*mInHeight+p] > m) {
							mIndex = depthOffset+q*mInHeight+p;
							m = x[mIndex];
						}
					}
				}
				yFlattened[resultIndex] = m;
				mIndices[resultIndex] = mIndex;
				++resultIndex;
			}
		}
	}
	mBackpropIndices[sampleNum] = mIndices;
	return arma::Cube<double>(yFlattened.begin(), mInHeight/mFieldHeight, mInWidth/mFieldWidth, mInDepth);
}

arma::field<arma::Cube<double>> MaxPool::feedForward(
		const arma::field<arma::Cube<double>>& xs) {
	mBackpropIndices = arma::field<arma::Col<unsigned int>>(xs.size());
	arma::field<arma::Cube<double>> ys(xs.size());
	for (unsigned int i = 0; i < xs.size(); ++i) {
		ys(i) = feedForward(xs(i), i);
	}
	return ys;
}

arma::Cube<double> MaxPool::backProp(const arma::Cube<double>& delta,
		unsigned int sampleNum) {
	arma::Cube<double> deltaExpanded(mInHeight, mInWidth, mInDepth, arma::fill::zeros);
	arma::Col<unsigned int> mIndices = mBackpropIndices(sampleNum);
	for (unsigned int i=0; i<delta.size(); ++i) {
		deltaExpanded[mIndices[i]] = delta[i];
	}
	return deltaExpanded;

}

arma::field<arma::Cube<double>> MaxPool::backProp(
		const arma::field<arma::Cube<double>>& deltas) {
	arma::field<arma::Cube<double>> dxs(deltas.size());
	for (unsigned int i = 0; i<deltas.size(); ++i) {
		dxs(i) = backProp(deltas(i), i);
	}
	return dxs;
}
