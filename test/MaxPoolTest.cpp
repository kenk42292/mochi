/*
 * MaxPoolTest.cpp
 *
 *  Created on: Jan 2, 2017
 *      Author: ken
 */

#include "MaxPoolTest.hpp"

MaxPoolTest::MaxPoolTest() {}

MaxPoolTest::~MaxPoolTest() {}

void MaxPoolTest::feedForwardTest1() {
	std::vector<unsigned int> inDim(3);
	inDim[0] = 2;
	inDim[1] = 6;
	inDim[2] = 6;
	std::vector<unsigned int> fieldDim(2);
	fieldDim[0] = 2;
	fieldDim[1] = 2;
	std::vector<unsigned int> outDim(3);
	outDim[0] = 2;
	outDim[1] = 3;
	outDim[2] = 3;
	MaxPool m(inDim, fieldDim, outDim);

	arma::Cube<double> in(6, 6, 2, arma::fill::randu);
	in(0, 0, 0)=4;
	in(1, 2, 0)=3;
	in(1, 5, 0)=2;
	in(2, 1, 0)=6;
	in(2, 3, 0)=5;
	in(3, 4, 0)=7;
	in(5, 0, 0)=2;
	in(4, 2, 0)=2;
	in(5, 5, 0)=5;

	in(1, 0, 1)=5;
	in(0, 2, 1)=2;
	in(0, 5, 1)=2;
	in(2, 1, 1)=6;
	in(3, 2, 1)=9;
	in(2, 5, 1)=4;
	in(5, 1, 1)=5;
	in(4, 3, 1)=6;
	in(4, 4, 1)=3;

	arma::field<arma::Cube<double>> xs(1);
	xs(0) = in;

	arma::Cube<double> out(3, 3, 2);
	out(0, 0, 0) = 4;
	out(0, 1, 0) = 3;
	out(0, 2, 0) = 2;
	out(1, 0, 0) = 6;
	out(1, 1, 0) = 5;
	out(1, 2, 0) = 7;
	out(2, 0, 0) = 2;
	out(2, 1, 0) = 2;
	out(2, 2, 0) = 5;

	out(0, 0, 1) = 5;
	out(0, 1, 1) = 2;
	out(0, 2, 1) = 2;
	out(1, 0, 1) = 6;
	out(1, 1, 1) = 9;
	out(1, 2, 1) = 4;
	out(2, 0, 1) = 5;
	out(2, 1, 1) = 6;
	out(2, 2, 1) = 3;

	arma::field<arma::Cube<double>> ys = m.feedForward(xs);
	arma::Cube<double> y = ys(0);

	ASSERT_EQUALM("N_slices mismatch between expected and actual output", out.n_slices, y.n_slices);
	ASSERT_EQUALM("N_rows mismatch between expected and actual output", out.n_rows, y.n_rows);
	ASSERT_EQUALM("N_cols mismatch between expected and actual output", out.n_cols, y.n_cols);
	for (unsigned int k=0; k<out.n_slices; ++k) {
		for (unsigned int i=0; i<out.n_rows; ++i) {
			for (unsigned int j=0; j<out.n_cols; ++j) {
				ASSERT_EQUALM("Mismatch between expected and actual output", out(i,j,k), y(i,j,k));
			}
		}
	}
}


void MaxPoolTest::feedForwardTest2() {
	std::vector<unsigned int> inDim(3);
	inDim[0] = 2;
	inDim[1] = 6;
	inDim[2] = 6;
	std::vector<unsigned int> fieldDim(2);
	fieldDim[0] = 2;
	fieldDim[1] = 2;
	std::vector<unsigned int> outDim(3);
	outDim[0] = 2;
	outDim[1] = 3;
	outDim[2] = 3;
	MaxPool m(inDim, fieldDim, outDim);

	arma::Cube<double> in(6, 6, 2, arma::fill::randu);
	in(0, 0, 0)=4;
	in(1, 2, 0)=3;
	in(1, 5, 0)=2;
	in(2, 1, 0)=6;
	in(2, 3, 0)=5;
	in(3, 4, 0)=7;
	in(5, 0, 0)=2;
	in(4, 2, 0)=2;
	in(5, 5, 0)=5;

	in(1, 0, 1)=5;
	in(0, 2, 1)=2;
	in(0, 5, 1)=2;
	in(2, 1, 1)=6;
	in(3, 2, 1)=9;
	in(2, 5, 1)=4;
	in(5, 1, 1)=5;
	in(4, 3, 1)=6;
	in(4, 4, 1)=3;

	arma::field<arma::Cube<double>> xs(2);
	xs(0) = in;
	xs(1) = arma::Cube<double>(in.begin(), 6, 6, 2);

	arma::Cube<double> out(3, 3, 2);
	out(0, 0, 0) = 4;
	out(0, 1, 0) = 3;
	out(0, 2, 0) = 2;
	out(1, 0, 0) = 6;
	out(1, 1, 0) = 5;
	out(1, 2, 0) = 7;
	out(2, 0, 0) = 2;
	out(2, 1, 0) = 2;
	out(2, 2, 0) = 5;

	out(0, 0, 1) = 5;
	out(0, 1, 1) = 2;
	out(0, 2, 1) = 2;
	out(1, 0, 1) = 6;
	out(1, 1, 1) = 9;
	out(1, 2, 1) = 4;
	out(2, 0, 1) = 5;
	out(2, 1, 1) = 6;
	out(2, 2, 1) = 3;

	arma::field<arma::Cube<double>> ys = m.feedForward(xs);
	arma::Cube<double> y1 = ys(0);
	arma::Cube<double> y2 = ys(1);

	ASSERT_EQUALM("N_slices mismatch between expected and actual output", out.n_slices, y1.n_slices);
	ASSERT_EQUALM("N_rows mismatch between expected and actual output", out.n_rows, y1.n_rows);
	ASSERT_EQUALM("N_cols mismatch between expected and actual output", out.n_cols, y1.n_cols);
	ASSERT_EQUALM("N_slices mismatch between expected and actual output", out.n_slices, y2.n_slices);
	ASSERT_EQUALM("N_rows mismatch between expected and actual output", out.n_rows, y2.n_rows);
	ASSERT_EQUALM("N_cols mismatch between expected and actual output", out.n_cols, y2.n_cols);
	for (unsigned int k=0; k<out.n_slices; ++k) {
		for (unsigned int i=0; i<out.n_rows; ++i) {
			for (unsigned int j=0; j<out.n_cols; ++j) {
				ASSERT_EQUALM("Mismatch between expected and actual output", out(i,j,k), y1(i,j,k));
				ASSERT_EQUALM("Mismatch between expected and actual output", out(i,j,k), y2(i,j,k));
			}
		}
	}
}


void MaxPoolTest::backPropTest1() {
	std::vector<unsigned int> inDim(3);
	inDim[0] = 2;
	inDim[1] = 6;
	inDim[2] = 6;
	std::vector<unsigned int> fieldDim(2);
	fieldDim[0] = 2;
	fieldDim[1] = 2;
	std::vector<unsigned int> outDim(3);
	outDim[0] = 2;
	outDim[1] = 3;
	outDim[2] = 3;
	MaxPool m(inDim, fieldDim, outDim);

	arma::Cube<double> in(6, 6, 2, arma::fill::randu);
	in(0, 0, 0)=4;
	in(1, 2, 0)=3;
	in(1, 5, 0)=2;
	in(2, 1, 0)=6;
	in(2, 3, 0)=5;
	in(3, 4, 0)=7;
	in(5, 0, 0)=2;
	in(4, 2, 0)=2;
	in(5, 5, 0)=5;

	in(1, 0, 1)=5;
	in(0, 2, 1)=2;
	in(0, 5, 1)=2;
	in(2, 1, 1)=6;
	in(3, 2, 1)=9;
	in(2, 5, 1)=4;
	in(5, 1, 1)=5;
	in(4, 3, 1)=6;
	in(4, 4, 1)=3;

	arma::field<arma::Cube<double>> xs(1);
	xs(0) = in;

	arma::Cube<double> out(3, 3, 2);
	out(0, 0, 0) = 4;
	out(0, 1, 0) = 3;
	out(0, 2, 0) = 2;
	out(1, 0, 0) = 6;
	out(1, 1, 0) = 5;
	out(1, 2, 0) = 7;
	out(2, 0, 0) = 2;
	out(2, 1, 0) = 2;
	out(2, 2, 0) = 5;

	out(0, 0, 1) = 5;
	out(0, 1, 1) = 2;
	out(0, 2, 1) = 2;
	out(1, 0, 1) = 6;
	out(1, 1, 1) = 9;
	out(1, 2, 1) = 4;
	out(2, 0, 1) = 5;
	out(2, 1, 1) = 6;
	out(2, 2, 1) = 3;

	arma::field<arma::Cube<double>> ys = m.feedForward(xs);
	arma::Cube<double> y = ys(0);

	arma::Cube<double> delta(3,3,2);
	delta(0, 0, 0) = 2;
	delta(0, 1, 0) = 4;
	delta(0, 2, 0) = 3;
	delta(1, 0, 0) = 6;
	delta(1, 1, 0) = 7;
	delta(1, 2, 0) = 5;
	delta(2, 0, 0) = 4;
	delta(2, 1, 0) = 1;
	delta(2, 2, 0) = 6;

	delta(0, 0, 1) = 2;
	delta(0, 1, 1) = 2;
	delta(0, 2, 1) = 4;
	delta(1, 0, 1) = 6;
	delta(1, 1, 1) = 3;
	delta(1, 2, 1) = 7;
	delta(2, 0, 1) = 1;
	delta(2, 1, 1) = 7;
	delta(2, 2, 1) = 7;

	arma::field<arma::Cube<double>> deltas(1);
	deltas(0) = delta;

	arma::field<arma::Cube<double>> dxs = m.backProp(deltas);
	arma::Cube<double> dx = dxs(0);


	arma::Cube<double> expected_dx(6,6,2, arma::fill::zeros);

	expected_dx(0, 0, 0)=2;
	expected_dx(1, 2, 0)=4;
	expected_dx(1, 5, 0)=3;
	expected_dx(2, 1, 0)=6;
	expected_dx(2, 3, 0)=7;
	expected_dx(3, 4, 0)=5;
	expected_dx(5, 0, 0)=4;
	expected_dx(4, 2, 0)=1;
	expected_dx(5, 5, 0)=6;

	expected_dx(1, 0, 1)=2;
	expected_dx(0, 2, 1)=2;
	expected_dx(0, 5, 1)=4;
	expected_dx(2, 1, 1)=6;
	expected_dx(3, 2, 1)=3;
	expected_dx(2, 5, 1)=7;
	expected_dx(5, 1, 1)=1;
	expected_dx(4, 3, 1)=7;
	expected_dx(4, 4, 1)=7;

	ASSERT_EQUALM("N_slices mismatch between expected and actual dx", expected_dx.n_slices, dx.n_slices);
	ASSERT_EQUALM("N_rows mismatch between expected and actual dx", expected_dx.n_rows, dx.n_rows);
	ASSERT_EQUALM("N_cols mismatch between expected and actual dx", expected_dx.n_cols, dx.n_cols);

	for (unsigned int k=0; k<dx.n_slices; ++k) {
		for (unsigned int i=0; i<dx.n_rows; ++i) {
			for (unsigned int j=0; j<dx.n_cols; ++j) {
				ASSERT_EQUALM("Mismatch between expected and actual dx", expected_dx(i,j,k), dx(i,j,k));
			}
		}
	}

}


void MaxPoolTest::backPropTest2() {
	std::vector<unsigned int> inDim(3);
	inDim[0] = 2;
	inDim[1] = 6;
	inDim[2] = 6;
	std::vector<unsigned int> fieldDim(2);
	fieldDim[0] = 2;
	fieldDim[1] = 2;
	std::vector<unsigned int> outDim(3);
	outDim[0] = 2;
	outDim[1] = 3;
	outDim[2] = 3;
	MaxPool m(inDim, fieldDim, outDim);

	arma::Cube<double> in(6, 6, 2, arma::fill::randu);
	in(0, 0, 0)=4;
	in(1, 2, 0)=3;
	in(1, 5, 0)=2;
	in(2, 1, 0)=6;
	in(2, 3, 0)=5;
	in(3, 4, 0)=7;
	in(5, 0, 0)=2;
	in(4, 2, 0)=2;
	in(5, 5, 0)=5;

	in(1, 0, 1)=5;
	in(0, 2, 1)=2;
	in(0, 5, 1)=2;
	in(2, 1, 1)=6;
	in(3, 2, 1)=9;
	in(2, 5, 1)=4;
	in(5, 1, 1)=5;
	in(4, 3, 1)=6;
	in(4, 4, 1)=3;

	arma::field<arma::Cube<double>> xs(2);
	xs(0) = in;
	xs(1) = arma::Cube<double>(in.begin(), 6, 6, 2);;

	arma::Cube<double> out(3, 3, 2);
	out(0, 0, 0) = 4;
	out(0, 1, 0) = 3;
	out(0, 2, 0) = 2;
	out(1, 0, 0) = 6;
	out(1, 1, 0) = 5;
	out(1, 2, 0) = 7;
	out(2, 0, 0) = 2;
	out(2, 1, 0) = 2;
	out(2, 2, 0) = 5;

	out(0, 0, 1) = 5;
	out(0, 1, 1) = 2;
	out(0, 2, 1) = 2;
	out(1, 0, 1) = 6;
	out(1, 1, 1) = 9;
	out(1, 2, 1) = 4;
	out(2, 0, 1) = 5;
	out(2, 1, 1) = 6;
	out(2, 2, 1) = 3;

	arma::field<arma::Cube<double>> ys = m.feedForward(xs);
	arma::Cube<double> y1 = ys(0);
	arma::Cube<double> y2 = ys(1);

	arma::Cube<double> delta(3,3,2);
	delta(0, 0, 0) = 2;
	delta(0, 1, 0) = 4;
	delta(0, 2, 0) = 3;
	delta(1, 0, 0) = 6;
	delta(1, 1, 0) = 7;
	delta(1, 2, 0) = 5;
	delta(2, 0, 0) = 4;
	delta(2, 1, 0) = 1;
	delta(2, 2, 0) = 6;

	delta(0, 0, 1) = 2;
	delta(0, 1, 1) = 2;
	delta(0, 2, 1) = 4;
	delta(1, 0, 1) = 6;
	delta(1, 1, 1) = 3;
	delta(1, 2, 1) = 7;
	delta(2, 0, 1) = 1;
	delta(2, 1, 1) = 7;
	delta(2, 2, 1) = 7;

	arma::field<arma::Cube<double>> deltas(2);
	deltas(0) = delta;
	deltas(1) = delta;

	arma::field<arma::Cube<double>> dxs = m.backProp(deltas);
	arma::Cube<double> dx1 = dxs(0);
	arma::Cube<double> dx2 = dxs(1);


	arma::Cube<double> expected_dx(6,6,2, arma::fill::zeros);

	expected_dx(0, 0, 0)=2;
	expected_dx(1, 2, 0)=4;
	expected_dx(1, 5, 0)=3;
	expected_dx(2, 1, 0)=6;
	expected_dx(2, 3, 0)=7;
	expected_dx(3, 4, 0)=5;
	expected_dx(5, 0, 0)=4;
	expected_dx(4, 2, 0)=1;
	expected_dx(5, 5, 0)=6;

	expected_dx(1, 0, 1)=2;
	expected_dx(0, 2, 1)=2;
	expected_dx(0, 5, 1)=4;
	expected_dx(2, 1, 1)=6;
	expected_dx(3, 2, 1)=3;
	expected_dx(2, 5, 1)=7;
	expected_dx(5, 1, 1)=1;
	expected_dx(4, 3, 1)=7;
	expected_dx(4, 4, 1)=7;

	ASSERT_EQUALM("N_slices mismatch between expected and actual dx", expected_dx.n_slices, dx1.n_slices);
	ASSERT_EQUALM("N_rows mismatch between expected and actual dx", expected_dx.n_rows, dx1.n_rows);
	ASSERT_EQUALM("N_cols mismatch between expected and actual dx", expected_dx.n_cols, dx1.n_cols);
	ASSERT_EQUALM("N_slices mismatch between expected and actual dx", expected_dx.n_slices, dx2.n_slices);
	ASSERT_EQUALM("N_rows mismatch between expected and actual dx", expected_dx.n_rows, dx2.n_rows);
	ASSERT_EQUALM("N_cols mismatch between expected and actual dx", expected_dx.n_cols, dx2.n_cols);

	for (unsigned int k=0; k<expected_dx.n_slices; ++k) {
		for (unsigned int i=0; i<expected_dx.n_rows; ++i) {
			for (unsigned int j=0; j<expected_dx.n_cols; ++j) {
				ASSERT_EQUALM("Mismatch between expected and actual dx", expected_dx(i,j,k), dx1(i,j,k));
				ASSERT_EQUALM("Mismatch between expected and actual dx", expected_dx(i,j,k), dx2(i,j,k));
			}
		}
	}

}
