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

/** Test that standard feedForward works for the valid config*/
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
	Optimizer* optimizer = new SGD(0.1);

	Convolutional c(1, inDim, numPatterns, patternDim, outDim, "valid", optimizer, 0.0);

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
	Optimizer* optimizer = new SGD(0.1);

	Convolutional c(1, inDim, numPatterns, patternDim, outDim, "valid", optimizer, 0.0);

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

/** Test that standard feedForward works for the "same" mode */
void ConvolutionalTest::feedForwardTest3() {
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
	outDim[1] = 4;
	outDim[2] = 4;
	Optimizer* optimizer = new SGD(0.1);

	Convolutional c(1, inDim, numPatterns, patternDim, outDim, "same", optimizer, 0.0);

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
	 *
	 *
     *  [[116, 171, 205, 151],
     *   [172, 238, 291, 194],
     *   [156, 247, 238, 160],
     *   [ 94, 137, 152,  80]]
     *
	 *  [[124, 192, 248, 156],
     *   [166, 276, 338, 209],
     *   [182, 295, 291, 177],
     *   [121, 180, 177, 133]]
     *
	 *  [[ 96, 148, 191, 108],
     *   [135, 227, 275, 180],
     *   [148, 238, 235, 151],
     *   [ 97, 159, 150, 117]]
	 *
	 * */
	double yA[] = { 116, 172, 156, 94, 171, 238, 247, 137, 205, 291, 238, 152, 151, 194, 160, 80,
					124, 166, 182, 121, 192, 276, 295, 180, 248, 338, 291, 177, 156, 209, 177, 133,
					96, 135, 148, 97, 148, 227, 238, 159, 191, 275, 235, 150, 108, 180, 151, 117};
	arma::Cube<double> expectedY(yA, 4, 4, 3);

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
	Optimizer* optimizer = new SGD(0.1);

	Convolutional c(1, inDim, numPatterns, patternDim, outDim, "valid", optimizer, 0.0);

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
	Optimizer* optimizer = new SGD(0.1);

	Convolutional c(1, inDim, numPatterns, patternDim, outDim, "valid", optimizer, 0.0);

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


/** Testing backProp for "same" mode */
void ConvolutionalTest::backPropTest3() {
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
	outDim[1] = 4;
	outDim[2] = 4;
	Optimizer* optimizer = new SGD(0.1);

	Convolutional c(1, inDim, numPatterns, patternDim, outDim, "same", optimizer, 0.0);

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
	 *
	 *
     *  [[116, 171, 205, 151],
     *   [172, 238, 291, 194],
     *   [156, 247, 238, 160],
     *   [ 94, 137, 152,  80]]
     *
	 *  [[124, 192, 248, 156],
     *   [166, 276, 338, 209],
     *   [182, 295, 291, 177],
     *   [121, 180, 177, 133]]
     *
	 *  [[ 96, 148, 191, 108],
     *   [135, 227, 275, 180],
     *   [148, 238, 235, 151],
     *   [ 97, 159, 150, 117]]
	 *
	 * */
	double yA[] = { 116, 172, 156, 94, 171, 238, 247, 137, 205, 291, 238, 152, 151, 194, 160, 80,
					124, 166, 182, 121, 192, 276, 295, 180, 248, 338, 291, 177, 156, 209, 177, 133,
					96, 135, 148, 97, 148, 227, 238, 159, 191, 275, 235, 150, 108, 180, 151, 117};
	arma::Cube<double> expectedY(yA, 4, 4, 3);

	arma::field<arma::Cube<double>> ys = c.feedForward(xs);
	arma::Cube<double> y = ys(0);

	/**
	 * [[2,4,1,6],
	 *  [5,3,6,2],
	 *  [2,5,6,1],
	 *  [1,1,7,3]]
	 * [[6,1,5,5],
	 *  [4,1,2,1],
	 *  [7,6,4,8],
	 *  [4,5,2,7]]
	 * [[3,4,8,8],
	 *  [2,5,1,2],
	 *  [5,5,3,4],
	 *  [1,7,8,9]]
	 */
	double deltaA[] = {2,5,2,1,4,3,5,1,1,6,6,7,6,2,1,3,
					   6,4,7,4,1,1,6,5,5,2,4,2,5,1,8,7,
					   3,2,5,1,4,5,5,7,8,1,3,8,8,2,4,9};
	arma::Cube<double> delta(deltaA, 4,4,3); //should properly be cast to 4x4x3 within conv. layer

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
	 * [[[ 91, 129, 159],
         [139, 207, 154],
         [130, 164, 127]],

        [[155, 204, 185],
         [187, 247, 176],
         [110, 175, 137]]]
	 *
	 * expected_dw2:
	 * [[[115, 196, 131],
         [147, 215, 161],
         [148, 202, 145]],

        [[149, 208, 148],
         [238, 317, 244],
         [108, 141, 118]]]
	 *
	 * expected_dw3:
	 * [[[113, 173, 132],
         [215, 254, 194],
         [161, 222, 155]],

        [[204, 264, 192],
         [293, 351, 246],
         [131, 136, 115]]]
	 * */

	double expected_dw1A[] = {91,139,130,129,207,164,159,154,127,155,187,110,204,247,175,185,176,137};
	arma::Cube<double> expected_dw1(expected_dw1A, 3, 3, 2);
	double expected_dw2A[] = {115,147,148,196,215,202,131,161,145,149,238,108,208,317,141,148,244,118};
	arma::Cube<double> expected_dw2(expected_dw2A, 3, 3, 2);
	double expected_dw3A[] = {113,215,161,173,254,222,132,194,155,204,293,131,264,351,136,192,246,115};
	arma::Cube<double> expected_dw3(expected_dw3A, 3, 3, 2);

	double expected_dbA[] = {55, 68, 75};
	arma::Cube<double> expected_db(expected_dbA, 1, 1, 3);

	/**
	 * expected_dx:
	 * [[[120, 238, 207, 213],
         [240, 411, 413, 347],
         [225, 442, 444, 334],
         [184, 341, 389, 294]],

	 *  [[123, 195, 193, 138],
         [234, 346, 350, 232],
         [249, 361, 393, 243],
         [159, 268, 326, 188]]]
	 */
	double expected_dxA[] = { 120,240,225,184,238,411,442,341,207,413,444,389,213,347,334,294,
								123,234,249,159,195,346,361,268,193,350,393,326,138,232,243,188 };
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
