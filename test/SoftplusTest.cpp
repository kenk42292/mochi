/*
 * SoftplusTest.cpp
 *
 *  Created on: Dec 30, 2016
 *      Author: ken
 */

#include "SoftplusTest.hpp"

SoftplusTest::SoftplusTest() {}

SoftplusTest::~SoftplusTest() {}

void SoftplusTest::feedForwardTest1() {
	Softplus s;
	arma::Cube<double> x1(1, 1, 3);
	x1(0, 0, 0) = 1;
	x1(0, 0, 1) = 5;
	x1(0, 0, 2) = 3;
	arma::Cube<double> x2(1, 1, 3);
	x2(0, 0, 0) = 4;
	x2(0, 0, 1) = 2;
	x2(0, 0, 2) = 6;
	arma::field<arma::Cube<double>> xs(2);
	xs(0) = x1;
	xs(1) = x2;
	arma::field<arma::Cube<double>> ys = s.feedForward(xs);
	arma::field<arma::Cube<double>> desired_ys(2);
	desired_ys(0) = arma::Cube<double>(1, 1, 3);
	// Desired output of x1: 1.31326169, 5.00671535, 3.04858735
	desired_ys(0)(0, 0, 0) = 1.31326169;
	desired_ys(0)(0, 0, 1) = 5.00671535;
	desired_ys(0)(0, 0, 2) = 3.04858735;
	// Desired output of x2: 4.01814993, 2.12692801, 6.00247569
	desired_ys(1) = arma::Cube<double>(1, 1, 3);
	desired_ys(1)(0, 0, 0) = 4.01814993;
	desired_ys(1)(0, 0, 1) = 2.12692801;
	desired_ys(1)(0, 0, 2) = 6.00247569;

	for (unsigned int i = 0; i < 2; ++i) {
		// Check for appropriat dimensions of output cubes
		ASSERT_EQUALM("dimensions mismatch", desired_ys(i).n_cols,
				ys(i).n_cols);
		ASSERT_EQUALM("dimensions mismatch", desired_ys(i).n_rows,
				ys(i).n_rows);
		ASSERT_EQUALM("dimensions mismatch", desired_ys(i).n_slices,
				ys(i).n_slices);
		for (unsigned int j = 0; j < 3; ++j) {
			ASSERT_EQUAL_DELTAM("softplus outputs differ from expected",
					desired_ys(i)(j), ys(i)(j), 0.0001);
		}
	}

	// Check that input of softplus was stored - needed for backprop
	ASSERT_EQUALM("stored mxs has wrong number of elements", 2, s.mxs.size());
	for (unsigned int i = 0; i < 2; ++i) {
		ASSERT_EQUALM("dimensions mismatch", xs(i).n_cols,
				s.mxs(i).n_cols);
		ASSERT_EQUALM("dimensions mismatch", xs(i).n_rows,
				s.mxs(i).n_rows);
		ASSERT_EQUALM("dimensions mismatch", xs(i).n_slices,
				s.mxs(i).n_slices);
	}
}

void SoftplusTest::feedForwardTest2() {
	Softplus s;
	arma::Cube<double> x1(2, 2, 2);
	x1(0, 0, 0) = 1;
	x1(0, 1, 0) = 2;
	x1(1, 0, 0) = 3;
	x1(1, 1, 0) = 4;
	x1(0, 0, 1) = 5;
	x1(0, 1, 1) = 6;
	x1(1, 0, 1) = 7;
	x1(1, 1, 1) = 8;
	arma::Cube<double> x2(2, 2, 2);
	x2(0, 0, 0) = 9;
	x2(0, 1, 0) = 8;
	x2(1, 0, 0) = 7;
	x2(1, 1, 0) = 6;
	x2(0, 0, 1) = 5;
	x2(0, 1, 1) = 4;
	x2(1, 0, 1) = 3;
	x2(1, 1, 1) = 2;
	arma::field<arma::Cube<double>> xs(2);
	xs(0) = x1;
	xs(1) = x2;
	arma::Cube<double> desired_y1(2, 2, 2);
	desired_y1(0, 0, 0) = 1.31326169;
	desired_y1(0, 1, 0) = 2.12692801;
	desired_y1(1, 0, 0) = 3.04858735;
	desired_y1(1, 1, 0) = 4.01814993;
	desired_y1(0, 0, 1) = 5.00671535;
	desired_y1(0, 1, 1) = 6.00247569;
	desired_y1(1, 0, 1) = 7.00091147;
	desired_y1(1, 1, 1) = 8.00033541;

	arma::Cube<double> desired_y2(2, 2, 2);
	desired_y2(0, 0, 0) = 9.0001234;
	desired_y2(0, 1, 0) = 8.00033541;
	desired_y2(1, 0, 0) = 7.00091147;
	desired_y2(1, 1, 0) = 6.00247569;
	desired_y2(0, 0, 1) = 5.00671535;
	desired_y2(0, 1, 1) = 4.01814993;
	desired_y2(1, 0, 1) = 3.04858735;
	desired_y2(1, 1, 1) = 2.12692801;
	arma::field<arma::Cube<double>> desired_ys(2);
	desired_ys(0) = desired_y1;
	desired_ys(1) = desired_y2;

	arma::field<arma::Cube<double>> ys = s.feedForward(xs);

	ASSERT_EQUALM("mismatch length btwn ys, desired_ys", ys.size(),
			desired_ys.size());
	for (unsigned int i = 0; i < 2; ++i) {
		ASSERT_EQUALM("size mismatch between ys, desired_ys slices",
				ys(i).n_slices, desired_ys(i).n_slices);
		ASSERT_EQUALM("size mismatch between ys, desired_ys rows", ys(i).n_rows,
				desired_ys(i).n_rows);
		ASSERT_EQUALM("size mismatch between ys, desired_ys cols", ys(i).n_cols,
				desired_ys(i).n_cols);
		for (unsigned int x = 0; x < 2; ++x) {
			for (unsigned int y = 0; y < 2; ++y) {
				for (unsigned int z = 0; z < 2; ++z) {
					ASSERT_EQUAL_DELTAM(
							"cube element mismatch, y and y_desired",
							ys(i)(x, y, z), desired_ys(i)(x, y, z), 0.0001);
				}
			}
		}
	}
}

void SoftplusTest::backPropTest1() {
	Softplus s;
	arma::Cube<double> x1(1, 1, 3);
	x1(0, 0, 0) = 1;
	x1(0, 0, 1) = 5;
	x1(0, 0, 2) = 3;
	arma::Cube<double> x2(1, 1, 3);
	x2(0, 0, 0) = 4;
	x2(0, 0, 1) = 2;
	x2(0, 0, 2) = 6;
	arma::field<arma::Cube<double>> xs(2);
	xs(0) = x1;
	xs(1) = x2;
	arma::field<arma::Cube<double>> ys = s.feedForward(xs);
	arma::field<arma::Cube<double>> deltas(2);
	deltas(0) = arma::Cube<double>(1, 1, 3);
	deltas(0)(0, 0, 0) = 0.2;
	deltas(0)(0, 0, 1) = 0.1;
	deltas(0)(0, 0, 2) = 0.3;
	deltas(1) = arma::Cube<double>(1, 1, 3);
	deltas(1)(0, 0, 0) = 0.1;
	deltas(1)(0, 0, 1) = 0.9;
	deltas(1)(0, 0, 2) = 0.1;

	arma::field<arma::Cube<double>> dxs = s.backProp(deltas);


	arma::Cube<double> desired_dx1(1, 1, 3);
	// 0.14621172, 0.09933071, 0.28577224
	desired_dx1(0, 0, 0) = 0.14621172;
	desired_dx1(0, 0, 1) = 0.09933071;
	desired_dx1(0, 0, 2) = 0.28577224;
	arma::Cube<double> desired_dx2(1, 1, 3);
	//0.09820138, 0.79271737, 0.09975274
	desired_dx2(0, 0, 0) = 0.09820138;
	desired_dx2(0, 0, 1) = 0.79271737;
	desired_dx2(0, 0, 2) = 0.09975274;

	arma::field<arma::Cube<double>> desired_dxs(2);
	desired_dxs(0) = desired_dx1;
	desired_dxs(1) = desired_dx2;

	for (unsigned int i = 0; i < 2; ++i) {
		for (unsigned int j = 0; j < 3; ++j) {
			ASSERT_EQUAL_DELTAM("Mismatch backpropogated deltas",
					desired_dxs(i)(0, 0, j), dxs(i)(0, 0, j), 0.0001);
		}
	}
}

//STOPPED RIGHT HERE!;
void SoftplusTest::backPropTest2() {
	Softplus s;
	arma::Cube<double> x1(2, 2, 2);
	x1(0, 0, 0) = 1;
	x1(0, 1, 0) = 2;
	x1(1, 0, 0) = 3;
	x1(1, 1, 0) = 4;
	x1(0, 0, 1) = 5;
	x1(0, 1, 1) = 6;
	x1(1, 0, 1) = 7;
	x1(1, 1, 1) = 8;
	arma::Cube<double> x2(2, 2, 2);
	x2(0, 0, 0) = 9;
	x2(0, 1, 0) = 8;
	x2(1, 0, 0) = 7;
	x2(1, 1, 0) = 6;
	x2(0, 0, 1) = 5;
	x2(0, 1, 1) = 4;
	x2(1, 0, 1) = 3;
	x2(1, 1, 1) = 2;
	arma::field<arma::Cube<double>> xs(2);
	xs(0) = x1;
	xs(1) = x2;

	arma::field<arma::Cube<double>> ys = s.feedForward(xs);

	// Testing what would happen if delta is not the same shape as x
	arma::Cube<double> delta1(1, 1, 8);
	delta1(0, 0, 0) = 1;
	delta1(0, 0, 1) = 2;
	delta1(0, 0, 2) = 5;
	delta1(0, 0, 3) = 4;
	delta1(0, 0, 4) = 7;
	delta1(0, 0, 5) = 3;
	delta1(0, 0, 6) = 2;
	delta1(0, 0, 7) = 7;
	arma::Cube<double> delta2(1, 1, 8);
	delta2(0, 0, 0) = 1;
	delta2(0, 0, 1) = 5;
	delta2(0, 0, 2) = 4;
	delta2(0, 0, 3) = 7;
	delta2(0, 0, 4) = 2;
	delta2(0, 0, 5) = 3;
	delta2(0, 0, 6) = 3;
	delta2(0, 0, 7) = 1;
	arma::field<arma::Cube<double>> deltas(2);
	deltas(0) = delta1;
	deltas(1) = delta2;

	arma::field<arma::Cube<double>> desired_dxs(2);
	arma::Cube<double> desired_dx1(2, 2, 2);
//	[ 0.73105858,  4.40398539],
//	        [ 1.90514825,  3.92805516]],
//
//	       [[ 6.95315004,  1.99505475],
//	        [ 2.99726685,  6.99765255]]
	desired_dx1(0, 0, 0) = 0.73105858;
	desired_dx1(0, 1, 0) = 4.40398539;
	desired_dx1(1, 0, 0) = 1.90514825;
	desired_dx1(1, 1, 0) = 3.92805516;
	desired_dx1(0, 0, 1) = 6.95315004;
	desired_dx1(0, 1, 1) = 1.99505475;
	desired_dx1(1, 0, 1) = 2.99726685;
	desired_dx1(1, 1, 1) = 6.99765255;

	arma::Cube<double> desired_dx2(2, 2, 2);
//	0.99987661,  3.9986586 ],
//	        [ 4.99544474,  6.98269164]],
//
//	       [[ 1.9866143 ,  2.94604137],
//	        [ 2.85772238,  0.88079708
	desired_dx2(0, 0, 0) = 0.99987661;
	desired_dx2(0, 1, 0) = 3.9986586;
	desired_dx2(1, 0, 0) = 4.99544474;
	desired_dx2(1, 1, 0) = 6.98269164;
	desired_dx2(0, 0, 1) = 1.9866143;
	desired_dx2(0, 1, 1) = 2.94604137;
	desired_dx2(1, 0, 1) = 2.85772238;
	desired_dx2(1, 1, 1) = 0.88079708;

	desired_dxs(0) = desired_dx1;
	desired_dxs(1) = desired_dx2;

	arma::field<arma::Cube<double>> dxs = s.backProp(deltas);

	for (unsigned int i = 0; i < 2; ++i) {
		ASSERT_EQUALM("mismatching n_slices: desired dx, dx",
				desired_dxs(i).n_slices, dxs(i).n_slices);
		ASSERT_EQUALM("mismatching n_rows: desired dx, dx",
				desired_dxs(i).n_rows, dxs(i).n_rows);
		ASSERT_EQUALM("mismatching n_cols: desired dx, dx",
				desired_dxs(i).n_cols, dxs(i).n_cols);
	}

	for (unsigned int p = 0; p < 2; ++p) {
		for (unsigned int i = 0; i < 2; ++i) {
			for (unsigned int j = 0; j < 2; ++j) {
				for (unsigned int k = 0; k < 2; ++k) {
					ASSERT_EQUAL_DELTAM("mismatch between desired dx and dx",
							desired_dxs(p)(i, j, k), dxs(p)(i, j, k), 0.00001);
				}
			}
		}
	}

}
