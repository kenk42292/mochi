/*
 * ConvolutionalTest.cpp
 *
 *  Created on: Jan 3, 2017
 *      Author: ken
 */

#include "ConvolutionalTest.hpp"

ConvolutionalTest::ConvolutionalTest() {
}

ConvolutionalTest::~ConvolutionalTest() {
}

arma::field<arma::Cube<double>> ConvolutionalTest::mockWeights() {
	double w1A[] = { 3, 2, 4, 1, 5, 6, 4, 3, 5, 2, 3, 4, 2, 3, 5, 4, 3, 5 };
	double w2A[] = { 1, 1, 4, 3, 5, 6, 7, 7, 6, 6, 5, 4, 4, 3, 2, 1, 4, 5 };
	double w3A[] = { 3, 3, 2, 4, 3, 4, 5, 5, 6, 4, 3, 4, 3, 2, 1, 2, 3, 4 };
	/*
	 *w1:
	 *[[3,1,4],
	 * [2,5,3],
	 * [4,6,5]],
	 *[[2,2,4],
	 * [3,3,3],
	 * [4,5,5]]
	 *w2:
	 *[[1,3,7],
	 * [1,5,7],
	 * [4,6,6]],
	 *[[6,4,1],
	 * [5,3,4],
	 * [4,2,5]]
	 *w3:
	 *[[3,4,5],
	 * [3,3,5],
	 * [2,4,6]],
	 *[[4,3,2],
	 * [3,2,3],
	 * [4,1,4]]
	 * */
	arma::Cube<double> w1(w1A, 3, 3, 2);
	arma::Cube<double> w2(w2A, 3, 3, 2);
	arma::Cube<double> w3(w3A, 3, 3, 2);

	arma::field<arma::Cube<double>> ws(3);
	ws(0) = w1;
	ws(1) = w2;
	ws(2) = w3;

	return ws;
}

arma::Cube<double> ConvolutionalTest::mockBias() {
	arma::Cube<double> b(1, 1, 3);
	b(0, 0, 0) = 3;
	b(0, 0, 1) = 2;
	b(0, 0, 2) = 5;
	return b;
}

/** Test that standard feedForward works */
void ConvolutionalTest::feedForwardTest1() {
	std::vector<unsigned int> inDim(3);
	inDim[0] = 2;
	inDim[1] = 4;
	inDim[2] = 4;
	unsigned int numPatterns = 3;
	std::vector<unsigned int> patternDim(3);
	patternDim[0] = 2;
	patternDim[1] = 3;
	patternDim[2] = 3;
	std::vector<unsigned int> outDim(3);
	outDim[0] = 3;
	outDim[1] = 2;
	outDim[2] = 2;
	Optimizer* optimizer = new GradientDescent(0.1);

	Convolutional c(inDim, numPatterns, patternDim, outDim, optimizer);

	c.mws = mockWeights();
	c.mbs = mockBias();

	/*
	 * INPUT:
	 *[[1,2,3,5],
	 * [4,3,6,7],
	 * [2,3,1,4],
	 * [5,3,6,2]],
	 *[[4,7,6,8],
	 * [3,3,1,1],
	 * [6,4,7,8],
	 * [2,3,5,1]]
	 */
	double xA[] = { 1, 4, 2, 5, 2, 3, 3, 3, 3, 6, 1, 6, 5, 7, 4, 2, 4, 3, 6, 2,
			7, 3, 4, 3, 6, 1, 7, 5, 8, 1, 8, 1 };
	arma::Cube<double> x(xA, 4, 4, 2);
	arma::field<arma::Cube<double>> xs(1);
	xs(0) = x;

	/*
	 * EXPECTED OUTPUT:
	 * [[238,291],
	 *  [247,238]]
	 * [[276,338],
	 *  [295,291]]
	 * [[227,275],
	 *  [238,235]]
	 * */
	double yA[] = { 238, 247, 291, 238, 276, 295, 338, 291, 227, 238, 275, 235 };
	arma::Cube<double> expectedY(yA, 2, 2, 3);

	arma::field<arma::Cube<double>> ys = c.feedForward(xs);
	arma::Cube<double> y = ys(0);

	ASSERT_EQUALM("expected and actual output mismatch - n_slices",
			expectedY.n_slices, y.n_slices);
	ASSERT_EQUALM("expected and actual output mismatch - n_rows",
			expectedY.n_rows, y.n_rows);
	ASSERT_EQUALM("expected and actual output mismatch - n_cols",
			expectedY.n_cols, y.n_cols);

	for (unsigned int k = 0; k < y.n_slices; ++k) {
		for (unsigned int i = 0; i < y.n_rows; ++i) {
			for (unsigned int j = 0; j < y.n_cols; ++j) {
				ASSERT_EQUAL_DELTAM("expected and actual output mismatch",
						expectedY(i, j, k), y(i, j, k), 0.00001);
			}
		}
	}

}

/** Testing when actual input dimensions don't match the promised input dimensions*/
void ConvolutionalTest::feedForwardTest2() {
	std::vector<unsigned int> inDim(3);
	inDim[0] = 2;
	inDim[1] = 4;
	inDim[2] = 4;
	unsigned int numPatterns = 3;
	std::vector<unsigned int> patternDim(3);
	patternDim[0] = 2;
	patternDim[1] = 3;
	patternDim[2] = 3;
	std::vector<unsigned int> outDim(3);
	outDim[0] = 3;
	outDim[1] = 2;
	outDim[2] = 2;
	Optimizer* optimizer = new GradientDescent(0.1);

	Convolutional c(inDim, numPatterns, patternDim, outDim, optimizer);

	c.mws = mockWeights();
	c.mbs = mockBias();

	/*
	 * INPUT:
	 *[[1,2,3,5],
	 * [4,3,6,7],
	 * [2,3,1,4],
	 * [5,3,6,2]],
	 *[[4,7,6,8],
	 * [3,3,1,1],
	 * [6,4,7,8],
	 * [2,3,5,1]]
	 */
	double xA[] = { 1, 4, 2, 5, 2, 3, 3, 3, 3, 6, 1, 6, 5, 7, 4, 2, 4, 3, 6, 2,
			7, 3, 4, 3, 6, 1, 7, 5, 8, 1, 8, 1 };
	arma::Cube<double> x(xA, 1, 1, 32);
	arma::field<arma::Cube<double>> xs(1);
	xs(0) = x;

	/*
	 * EXPECTED OUTPUT:
	 * [[238,291],
	 *  [247,238]]
	 * [[276,338],
	 *  [295,291]]
	 * [[227,275],
	 *  [238,235]]
	 * */
	double yA[] = { 238, 247, 291, 238, 276, 295, 338, 291, 227, 238, 275, 235 };
	arma::Cube<double> expectedY(yA, 2, 2, 3);

	arma::field<arma::Cube<double>> ys = c.feedForward(xs);
	arma::Cube<double> y = ys(0);

	ASSERT_EQUALM("expected and actual output mismatch - n_slices",
			expectedY.n_slices, y.n_slices);
	ASSERT_EQUALM("expected and actual output mismatch - n_rows",
			expectedY.n_rows, y.n_rows);
	ASSERT_EQUALM("expected and actual output mismatch - n_cols",
			expectedY.n_cols, y.n_cols);

	for (unsigned int k = 0; k < y.n_slices; ++k) {
		for (unsigned int i = 0; i < y.n_rows; ++i) {
			for (unsigned int j = 0; j < y.n_cols; ++j) {
				ASSERT_EQUAL_DELTAM("expected and actual output mismatch",
						expectedY(i, j, k), y(i, j, k), 0.00001);
			}
		}
	}
}
