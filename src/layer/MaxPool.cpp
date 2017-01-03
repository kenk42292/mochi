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

MaxPool::~MaxPool() {
}

arma::Mat<double> maxPoolNaive(const arma::Mat<double>& input) {
	arma::Mat<double> backPropMatrix(input.n_rows, input.n_cols,
			arma::fill::zeros);
	arma::Mat<double> y(input.n_rows / 2, input.n_cols / 2);
	for (unsigned int i = 0; i < input.n_rows; i += 2) {
		for (unsigned int j = 0; j < input.n_cols; j += 2) {
			const arma::Mat<double>& subMat = input.submat(i, j, i + 1, j + 1);
			auto indices = arma::ind2sub(
					arma::size(subMat.n_rows, subMat.n_cols),
					subMat.index_max());
			backPropMatrix(i + indices(0), j + indices(1)) = 1;
			y(i / 2, j / 2) = subMat.max();
		}
	}
	return y;
}

arma::Cube<double> MaxPool::feedForward(const arma::Cube<double>& x,
		unsigned int sampleNum) {
	arma::Cube<double> backPropCube(x.n_rows, x.n_cols, x.n_slices,
			arma::fill::zeros);
	arma::Cube<double> y(x.n_rows / mFieldHeight, x.n_cols / mFieldWidth,
			x.n_slices);
	for (unsigned int k = 0; k < x.n_slices; ++k) {
		for (unsigned int i = 0; i < x.n_rows; i += mFieldHeight) {
			for (unsigned int j = 0; j < x.n_cols; j += mFieldWidth) {
				const arma::Cube<double>& subCube = x.subcube(i, j, k,
						i + mFieldHeight - 1, j + mFieldWidth - 1, k);
				auto indices = arma::ind2sub(
						arma::size(subCube.n_rows, subCube.n_cols,
								subCube.n_slices), subCube.index_max());
				backPropCube(i + indices(0), j + indices(1), k + indices(2)) =
						1;
				y(i / mFieldHeight, j / mFieldWidth, k) = subCube.max();
			}
		}
	}
	mBackPropCubes(sampleNum) = backPropCube;
	return y;
}

arma::field<arma::Cube<double>> MaxPool::feedForward(
		const arma::field<arma::Cube<double>>& xs) {
	mBackPropCubes = arma::field<arma::Cube<double>>(xs.size());
	arma::field<arma::Cube<double>> ys(xs.size());
	for (unsigned int i = 0; i < xs.size(); ++i) {
		const arma::Cube<double>& x = xs(i);
		ys(i) = feedForward(x, i);
	}
	return ys;
}

arma::Cube<double> MaxPool::backProp(const arma::Cube<double>& delta,
		unsigned int sampleNum) {
	arma::Cube<double> deltaExpanded(mInHeight, mInWidth, mInDepth);
	for (unsigned int k = 0; k < delta.n_slices; ++k) {
		for (unsigned int i = 0; i < delta.n_rows; ++i) {
			for (unsigned int j = 0; j < delta.n_cols; ++j) {
				for (unsigned int y=mFieldHeight*i; y<mFieldHeight*i+mFieldHeight; ++y) {
					for (unsigned int x=mFieldWidth*j; x<mFieldWidth*j+mFieldWidth; ++x) {
						deltaExpanded(y,x,k) = delta(i,j,k);
					}
				}
			}
		}
	}
	return deltaExpanded%mBackPropCubes(sampleNum);
}

arma::field<arma::Cube<double>> MaxPool::backProp(
		const arma::field<arma::Cube<double>>& deltas) {
	arma::field<arma::Cube<double>> dxs(deltas.size());
	for (unsigned int i = 0; i<deltas.size(); ++i) {
		dxs(i) = backProp(deltas(i), i);
	}
	return dxs;
}
