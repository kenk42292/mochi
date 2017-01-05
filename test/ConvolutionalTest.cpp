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

	/* Make sure xs are properly stored */
	/*for (unsigned int p=0; p<xs.size(); ++p) {
		ASSERT_EQUALM("Dim. mismatch btwn stored and actual xs - slices", xs(p).n_slices, c.mxs(p).n_slices);
		ASSERT_EQUALM("Dim. mismatch btwn stored and actual xs - rows", xs(p).n_rows, c.mxs(p).n_rows);
		ASSERT_EQUALM("Dim. mismatch btwn stored and actual xs - cols", xs(p).n_cols, c.mxs(p).n_cols);
		for (unsigned int k=0; k<xs(p).n_slices; ++k) {
			for (unsigned int i=0; i<xs(p).n_rows; ++i) {
				for (unsigned int j=0; j<xs(p).n_cols; ++j) {
					ASSERT_EQUALM("Mismatch btwn stored and actual xs", xs(p)(i,j,k), c.mxs(p)(i,j,k));
				}
			}
		}
	}*/

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

void ConvolutionalTest::backPropTest1() {
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

	/**
	 * [[2,4],
	 *  [5,3]]
	 * [[6,1],
	 *  [4,1]]
	 * [[3,4],
	 *  [2,5]]
	 */
	double deltaA[] = { 2, 5, 4, 3, 6, 4, 1, 1, 3, 2, 4, 5 };
	arma::Cube<double> delta(deltaA, 2, 2, 3);

	arma::field<arma::Cube<double>> deltas(1);
	deltas(0) = delta;

	arma::field<arma::Cube<double>> grads = c.getGrads(deltas);

	arma::Cube<double> dw1 = grads(0);
	arma::Cube<double> dw2 = grads(1);
	arma::Cube<double> dw3 = grads(2);
	arma::Cube<double> db = grads(3);
	arma::Cube<double> dx = grads(4);

	/**
	 * expected_dw1:
	 * [[[39, 49, 77],
	 *   [39, 48, 57],
	 *   [50, 43, 54]],
	 *  [[60, 56, 52],
	 *   [60, 51, 65],
	 *   [47, 66, 74]]]
	 *
	 * expected_dw2:
	 * [[[27, 33, 54],
	 *   [38, 37, 51],
	 *   [38, 37, 36]],
	 *  [[46, 61, 49],
	 *   [49, 42, 43],
	 *   [51, 48, 71]]])
	 *
	 * expected_dw3:
	 * [[[34, 54, 76],
	 *   [43, 44, 68],
	 *   [43, 49, 41]],
	 *  [[61, 56, 57],
	 *   [53, 56, 61],
	 *   [53, 71, 68]]]
	 * */

	double expected_dw1A[] = { 39, 39, 50, 49, 48, 43, 77, 57, 54, 60, 60, 47,
			56, 51, 66, 52, 65, 74 };
	arma::Cube<double> expected_dw1(expected_dw1A, 3, 3, 2);
	double expected_dw2A[] = { 27, 38, 38, 33, 37, 37, 54, 51, 36, 46, 49, 51,
			61, 42, 48, 49, 43, 71 };
	arma::Cube<double> expected_dw2(expected_dw2A, 3, 3, 2);
	double expected_dw3A[] = { 34, 43, 43, 54, 44, 49, 76, 68, 41, 61, 53, 53,
			56, 56, 71, 57, 61, 68 };
	arma::Cube<double> expected_dw3(expected_dw3A, 3, 3, 2);

	double expected_dbA[] = { 14, 12, 14 };
	arma::Cube<double> expected_db(expected_dbA, 1, 1, 3);

	/**
	 * expected_dx:
	 * [[[ 21,  57,  88,  43],
	 *   [ 44, 120, 184,  83],
	 *   [ 58, 161, 198,  91],
	 *   [ 40,  88, 105,  51]],
	 *  [[ 52,  67,  44,  25],
	 *   [ 87, 123, 115,  51],
	 *   [ 85, 121, 137,  69],
	 *   [ 44,  71,  75,  40]]]
	 */
	double expected_dxA[] = { 21, 44, 58, 40, 57, 120, 161, 88, 88, 184, 198,
			105, 43, 83, 91, 51, 52, 87, 85, 44, 67, 123, 121, 71, 44, 115, 137,
			75, 25, 51, 69, 40 };
	arma::Cube<double> expected_dx(expected_dxA, 4, 4, 2);

	arma::field<arma::Cube<double>> expected_grads(5);
	expected_grads(0) = expected_dw1;
	expected_grads(1) = expected_dw2;
	expected_grads(2) = expected_dw3;
	expected_grads(3) = expected_db;
	expected_grads(4) = expected_dx;

	for (unsigned int g = 0; g < 5; ++g) {
		const arma::Cube<double>& expected = expected_grads(g);
		const arma::Cube<double>& actual = grads(g);
		ASSERT_EQUALM(
				"Mismatch between expected and actual - n_slices, grad index: "
						+ std::to_string(g), expected.n_slices,
				actual.n_slices);
		ASSERT_EQUALM("Mismatch between expected and actual - n_rows, grad index: " + std::to_string(g),
				expected.n_rows, actual.n_rows);
		ASSERT_EQUALM("Mismatch between expected and actual - n_cols, grad index: " + std::to_string(g),
				expected.n_cols, actual.n_cols);
		for (unsigned int k = 0; k < expected.n_slices; ++k) {
			for (unsigned int i = 0; i < expected.n_rows; ++i) {
				for (unsigned int j = 0; j < expected.n_cols; ++j) {
					ASSERT_EQUAL_DELTAM(
							"Mismatch between expected and actual, grad index: "
									+ std::to_string(g), expected(i, j, k),
							actual(i, j, k), 0.000001);
				}
			}
		}

	}

}

/** What happens when delta is shaped like a Col instead of the expected Cube? */
void ConvolutionalTest::backPropTest2() {
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

	/**
	 * [[2,4],
	 *  [5,3]]
	 * [[6,1],
	 *  [4,1]]
	 * [[3,4],
	 *  [2,5]]
	 */
	double deltaA[] = { 2, 5, 4, 3, 6, 4, 1, 1, 3, 2, 4, 5 };
	arma::Cube<double> delta(deltaA, 1, 1, 12);

	arma::field<arma::Cube<double>> deltas(1);
	deltas(0) = delta;

	arma::field<arma::Cube<double>> grads = c.getGrads(deltas);
	arma::Cube<double> dw1 = grads(0);
	arma::Cube<double> dw2 = grads(1);
	arma::Cube<double> dw3 = grads(2);
	arma::Cube<double> db = grads(3);
	arma::Cube<double> dx = grads(4);

	/**
	 * expected_dw1:
	 * [[[39, 49, 77],
	 *   [39, 48, 57],
	 *   [50, 43, 54]],
	 *  [[60, 56, 52],
	 *   [60, 51, 65],
	 *   [47, 66, 74]]]
	 *
	 * expected_dw2:
	 * [[[27, 33, 54],
	 *   [38, 37, 51],
	 *   [38, 37, 36]],
	 *  [[46, 61, 49],
	 *   [49, 42, 43],
	 *   [51, 48, 71]]])
	 *
	 * expected_dw3:
	 * [[[34, 54, 76],
	 *   [43, 44, 68],
	 *   [43, 49, 41]],
	 *  [[61, 56, 57],
	 *   [53, 56, 61],
	 *   [53, 71, 68]]]
	 * */

	double expected_dw1A[] = { 39, 39, 50, 49, 48, 43, 77, 57, 54, 60, 60, 47,
			56, 51, 66, 52, 65, 74 };
	arma::Cube<double> expected_dw1(expected_dw1A, 3, 3, 2);
	double expected_dw2A[] = { 27, 38, 38, 33, 37, 37, 54, 51, 36, 46, 49, 51,
			61, 42, 48, 49, 43, 71 };
	arma::Cube<double> expected_dw2(expected_dw2A, 3, 3, 2);
	double expected_dw3A[] = { 34, 43, 43, 54, 44, 49, 76, 68, 41, 61, 53, 53,
			56, 56, 71, 57, 61, 68 };
	arma::Cube<double> expected_dw3(expected_dw3A, 3, 3, 2);

	double expected_dbA[] = { 14, 12, 14 };
	arma::Cube<double> expected_db(expected_dbA, 1, 1, 3);

	/**
	 * expected_dx:
	 * [[[ 21,  57,  88,  43],
	 *   [ 44, 120, 184,  83],
	 *   [ 58, 161, 198,  91],
	 *   [ 40,  88, 105,  51]],
	 *  [[ 52,  67,  44,  25],
	 *   [ 87, 123, 115,  51],
	 *   [ 85, 121, 137,  69],
	 *   [ 44,  71,  75,  40]]]
	 */
	double expected_dxA[] = { 21, 44, 58, 40, 57, 120, 161, 88, 88, 184, 198,
			105, 43, 83, 91, 51, 52, 87, 85, 44, 67, 123, 121, 71, 44, 115, 137,
			75, 25, 51, 69, 40 };
	arma::Cube<double> expected_dx(expected_dxA, 4, 4, 2);

	arma::field<arma::Cube<double>> expected_grads(5);
	expected_grads(0) = expected_dw1;
	expected_grads(1) = expected_dw2;
	expected_grads(2) = expected_dw3;
	expected_grads(3) = expected_db;
	expected_grads(4) = expected_dx;

	for (unsigned int g = 0; g < 5; ++g) {
		const arma::Cube<double>& expected = expected_grads(g);
		const arma::Cube<double>& actual = grads(g);
		ASSERT_EQUALM(
				"Mismatch between expected and actual - n_slices, grad index: "
						+ std::to_string(g), expected.n_slices,
				actual.n_slices);
		ASSERT_EQUALM("Mismatch between expected and actual - n_rows, grad index: " + std::to_string(g),
				expected.n_rows, actual.n_rows);
		ASSERT_EQUALM("Mismatch between expected and actual - n_cols, grad index: " + std::to_string(g),
				expected.n_cols, actual.n_cols);
		for (unsigned int k = 0; k < expected.n_slices; ++k) {
			for (unsigned int i = 0; i < expected.n_rows; ++i) {
				for (unsigned int j = 0; j < expected.n_cols; ++j) {
					ASSERT_EQUAL_DELTAM(
							"Mismatch between expected and actual, grad index: "
									+ std::to_string(g), expected(i, j, k),
							actual(i, j, k), 0.000001);
				}
			}
		}

	}

}

