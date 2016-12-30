/*
 * SigmoidTest.cpp
 *
 *  Created on: Dec 29, 2016
 *      Author: ken
 */

#include "SigmoidTest.hpp"

SigmoidTest::SigmoidTest() {
}

SigmoidTest::~SigmoidTest() {
}

void SigmoidTest::feedForwardTest1() {
	Sigmoid s;
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
	// Desired output of x1: 0.73105858,  0.99330715,  0.95257413
	desired_ys(0)(0, 0, 0) = 0.73105858;
	desired_ys(0)(0, 0, 1) = 0.99330715;
	desired_ys(0)(0, 0, 2) = 0.95257413;
	// Desired output of x2: 0.98201379,  0.88079708,  0.99752738
	desired_ys(1) = arma::Cube<double>(1, 1, 3);
	desired_ys(1)(0, 0, 0) = 0.98201379;
	desired_ys(1)(0, 0, 1) = 0.88079708;
	desired_ys(1)(0, 0, 2) = 0.99752738;

	for (unsigned int i = 0; i < 2; ++i) {
		// Check for appropriat dimensions of output cubes
		ASSERT_EQUALM("dimensions mismatch", desired_ys(i).n_cols,
				ys(i).n_cols);
		ASSERT_EQUALM("dimensions mismatch", desired_ys(i).n_rows,
				ys(i).n_rows);
		ASSERT_EQUALM("dimensions mismatch", desired_ys(i).n_slices,
				ys(i).n_slices);
		for (unsigned int j = 0; j < 3; ++j) {
			ASSERT_EQUAL_DELTAM("sigmoid outputs differ from expected",
					desired_ys(i)(j), ys(i)(j), 0.0001);
		}
	}

	// Check that output of sigmoid was stored
	ASSERT_EQUALM("stored mys has wrong number of elements", 2, s.mYs.size());
	for (unsigned int i = 0; i < 2; ++i) {
		ASSERT_EQUALM("dimensions mismatch", desired_ys(i).n_cols,
				s.mYs(i).n_cols);
		ASSERT_EQUALM("dimensions mismatch", desired_ys(i).n_rows,
				s.mYs(i).n_rows);
		ASSERT_EQUALM("dimensions mismatch", desired_ys(i).n_slices,
				s.mYs(i).n_slices);
		for (unsigned int j = 0; j < 3; ++j) {
			ASSERT_EQUAL_DELTAM("sigmoid outputs differ from expected",
					desired_ys(i)(j), s.mYs(i)(j), 0.0001);
		}
	}
}

void SigmoidTest::feedForwardTest2() {
	Sigmoid s;
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
	//0.73105858,  0.88079708,  0.95257413,  0.98201379,  0.99330715, 0.99752738,  0.99908895,  0.99966465
	arma::Cube<double> desired_y1(2, 2, 2);
	desired_y1(0, 0, 0) = 0.73105858;
	desired_y1(0, 1, 0) = 0.88079708;
	desired_y1(1, 0, 0) = 0.95257413;
	desired_y1(1, 1, 0) = 0.98201379;
	desired_y1(0, 0, 1) = 0.99330715;
	desired_y1(0, 1, 1) = 0.99752738;
	desired_y1(1, 0, 1) = 0.99908895;
	desired_y1(1, 1, 1) = 0.99966465;
	// 0.99987661,  0.99966465,  0.99908895,  0.99752738,  0.99330715, 0.98201379,  0.95257413,  0.88079708
	arma::Cube<double> desired_y2(2, 2, 2);
	desired_y2(0, 0, 0) = 0.99987661;
	desired_y2(0, 1, 0) = 0.99966465;
	desired_y2(1, 0, 0) = 0.99908895;
	desired_y2(1, 1, 0) = 0.99752738;
	desired_y2(0, 0, 1) = 0.99330715;
	desired_y2(0, 1, 1) = 0.98201379;
	desired_y2(1, 0, 1) = 0.95257413;
	desired_y2(1, 1, 1) = 0.88079708;
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

void SigmoidTest::backPropTest1() {
	Sigmoid s;
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
	// 0.03932239, 0.00066481, 0.013553
	desired_dx1(0, 0, 0) = 0.03932239;
	desired_dx1(0, 0, 1) = 0.00066481;
	desired_dx1(0, 0, 2) = 0.013553;
	arma::Cube<double> desired_dx2(1, 1, 3);
	//0.00176627, 0.09449423, 0.00024665
	desired_dx2(0, 0, 0) = 0.00176627;
	desired_dx2(0, 0, 1) = 0.09449423;
	desired_dx2(0, 0, 2) = 0.00024665;

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

void SigmoidTest::backPropTest2() {
	Sigmoid s;
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
	desired_dx1(0, 0, 0) = 0.19661193;
	desired_dx1(0, 1, 0) = 0.52496793;
	desired_dx1(1, 0, 0) = 0.09035332;
	desired_dx1(1, 1, 0) = 0.07065082;
	desired_dx1(0, 0, 1) = 0.0465364;
	desired_dx1(0, 1, 1) = 0.00493302;
	desired_dx1(1, 0, 1) = 0.00273066;
	desired_dx1(1, 1, 1) = 0.00234666;

	arma::Cube<double> desired_dx2(2, 2, 2);
	desired_dx2(0, 0, 0) = 1.23379350e-04;
	desired_dx2(0, 1, 0) = 1.34095068e-03;
	desired_dx2(1, 0, 0) = 4.55110590e-03;
	desired_dx2(1, 1, 0) = 1.72655650e-02;
	desired_dx2(0, 0, 1) = 1.32961133e-02;
	desired_dx2(0, 1, 1) = 5.29881186e-02;
	desired_dx2(1, 0, 1) = 1.35529979e-01;
	desired_dx2(1, 1, 1) = 1.04993585e-01;

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

