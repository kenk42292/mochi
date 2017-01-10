/*
 * MaxPool.hpp
 *
 *  Created on: Jan 2, 2017
 *      Author: ken
 */

#ifndef LAYER_MAXPOOL_HPP_
#define LAYER_MAXPOOL_HPP_

#include "Layer.hpp"

class MaxPool: public Layer {
private:
	unsigned int mInDepth;
	unsigned int mInHeight;
	unsigned int mInWidth;
	unsigned int mFieldHeight;
	unsigned int mFieldWidth;
	unsigned int mOutDepth;
	unsigned int mOutHeight;
	unsigned int mOutWidth;
	arma::field<arma::Col<unsigned int>> mBackpropIndices;
	friend class MaxPoolTest;
public:
	MaxPool(std::vector<unsigned int> inputDim,
			std::vector<unsigned int> fieldDim,
			std::vector<unsigned int> outputDim);
	virtual ~MaxPool();

	arma::Cube<double> feedForward(const arma::Cube<double>& x, unsigned int sampleNum);
	arma::field<arma::Cube<double>> feedForward(
			const arma::field<arma::Cube<double>>& xs);
	arma::Cube<double> backProp(const arma::Cube<double>& delta, unsigned int sampleNum);
	arma::field<arma::Cube<double>> backProp(
			const arma::field<arma::Cube<double>>& deltas);
};

#endif /* LAYER_MAXPOOL_HPP_ */
