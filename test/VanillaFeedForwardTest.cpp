/*
 * VanillaFeedForwardTest.cpp
 *
 *  Created on: Dec 28, 2016
 *      Author: ken
 */

#include "../test/VanillaFeedForwardTest.hpp"

void VanillaFeedForwardTest::feedForwardTest1() {
	Optimizer* optimizer = new GradientDescent(0.1);
	VanillaFeedForward vff(1, 3, 2, optimizer, 0.0);
	arma::Cube<double> w(2, 3, 1);
	for (unsigned int i = 0; i < 2; ++i) {
		for (unsigned int j = 0; j < 3; ++j) {
			w(i, j, 0) = 3 * i + j;
		}
	}
	arma::Cube<double> b(1, 1, 2);
	b(0, 0, 0) = 3;
	b(0, 0, 1) = 5;
	vff.mW = w;
	vff.mB = b;
	arma::Cube<double> x(1, 1, 3);
	x(0, 0, 0) = 2;
	x(0, 0, 1) = 4;
	x(0, 0, 2) = 6;
	arma::field<arma::Cube<double>> input(1);
	input(0) = x;
	arma::Cube<double> desiredOutput(1, 1, 2);
	desiredOutput(0, 0, 0) = 19;
	desiredOutput(0, 0, 1) = 57;
	arma::field<arma::Cube<double>> outputs = vff.feedForward(input);
	arma::Cube<double> output = outputs(0);

	//Make sure input is stored in vff as mxs
	for (unsigned int i=0; i<3; ++i) {
		ASSERT_EQUALM("vff stored mxs and input mismatch: ", x(i), vff.mxs(0)(i));
	}

	ASSERT_EQUAL(2, output.n_slices);
	ASSERT_EQUAL(1, output.n_rows);
	ASSERT_EQUAL(1, output.n_cols);
	for (unsigned int i = 0; i < 2; ++i) {
		ASSERT_EQUALM("Desired output for vff feedforward mismatch",
				desiredOutput(0, 0, i), output(0, 0, i));
	}
}

void VanillaFeedForwardTest::feedForwardTest2() {
	Optimizer* optimizer = new GradientDescent(0.1);
	VanillaFeedForward vff(1, 8, 4, optimizer, 0.0);
	arma::Cube<double> w(4, 8, 2);
	for (unsigned int i = 0; i < 4; ++i) {
		for (unsigned int j = 0; j < 8; ++j) {
			w(i, j, 0) = 8 * i + j;
		}
	}
	arma::Cube<double> b(1, 1, 4);
	b(0, 0, 0) = 3;
	b(0, 0, 1) = 5;
	b(0, 0, 2) = 1;
	b(0, 0, 3) = 9;
	vff.mW = w;
	vff.mB = b;
	arma::Cube<double> x(2, 2, 2);
	for (unsigned int k = 0; k < 2; ++k) {
		for (unsigned int i = 0; i < 2; ++i) {
			for (unsigned int j = 0; j < 2; ++j) {
				x(i, j, k) = 4 * k + 2 * i + j;
			}
		}
	}
	arma::field<arma::Cube<double>> input(1);
	input(0) = x;
	arma::Cube<double> desiredOutput(1, 1, 4);
	desiredOutput(0, 0, 0) = 141;
	desiredOutput(0, 0, 1) = 367;
	desiredOutput(0, 0, 2) = 587;
	desiredOutput(0, 0, 3) = 819;
	arma::field<arma::Cube<double>> outputs = vff.feedForward(input);
	arma::Cube<double> output = outputs(0);
	ASSERT_EQUAL(4, output.n_slices);
	ASSERT_EQUAL(1, output.n_rows);
	ASSERT_EQUAL(1, output.n_cols);
	for (unsigned int i = 0; i < 4; ++i) {
		ASSERT_EQUALM("Desired output for vff feedforward mismatch",
				output(0, 0, i), desiredOutput(0, 0, i));
	}
}

void VanillaFeedForwardTest::backPropTest1() {

	Optimizer* optimizer = new GradientDescent(0.1);
	VanillaFeedForward vff(1, 3, 2, optimizer, 0.0);
	arma::Cube<double> w(2, 3, 1);
	for (unsigned int i = 0; i < 2; ++i) {
		for (unsigned int j = 0; j < 3; ++j) {
			w(i, j, 0) = 3 * i + j;
		}
	}
	arma::Cube<double> b(1, 1, 2);
	b(0, 0, 0) = 3;
	b(0, 0, 1) = 5;
	vff.mW = w;
	vff.mB = b;
	arma::Cube<double> x(1, 1, 3);
	x(0, 0, 0) = 2;
	x(0, 0, 1) = 4;
	x(0, 0, 2) = 6;
	arma::field<arma::Cube<double>> input(1);
	input(0) = x;
	arma::Cube<double> label(1, 1, 2);
	label(0, 0, 0) = 17;
	label(0, 0, 1) = 70;
	arma::field<arma::Cube<double>> outputs = vff.feedForward(input); //output is [19, 57] cube
	arma::Cube<double> output = outputs(0);

	arma::Cube<double> delta = output - label; //Just pretend such a delta is returned: [2, -13]
	arma::Cube<double> desired_dx(1, 1, 3);
	desired_dx(0, 0, 0) = -39;
	desired_dx(0, 0, 1) = -50;
	desired_dx(0, 0, 2) = -61;
	arma::Cube<double> desired_dw(2, 3, 1);
	desired_dw(0, 0, 0) = 4;
	desired_dw(0, 1, 0) = 8;
	desired_dw(0, 2, 0) = 12;
	desired_dw(1, 0, 0) = -26;
	desired_dw(1, 1, 0) = -52;
	desired_dw(1, 2, 0) = -78;
	arma::Cube<double> desired_db(1, 1, 2);
	desired_db(0, 0, 0) = 2;
	desired_db(0, 0, 1) = -13;

	arma::field<arma::Cube<double>> deltas(1);
	deltas(0) = delta;
	arma::field<arma::Cube<double>> grads = vff.getGrads(deltas);

	arma::Cube<double> dw = grads(0);
	arma::Cube<double> db = grads(1);
	arma::Cube<double> dx = grads(2);

	/* Check w gradients */
	for (unsigned int i = 0; i < 2; ++i) {
		for (unsigned int j = 0; j < 3; ++j) {
			ASSERT_EQUALM(
					"Wrong dw for entry (" + std::to_string(i) + ","
							+ std::to_string(j) + ")", desired_dw(i, j, 0),
					dw(i, j, 0));
		}
	}

	/* Check b gradients */
	for (unsigned int i=0; i<2; ++i) {
		ASSERT_EQUALM("Wrong db for entry " + std::to_string(i) , desired_db(0, 0, i), db(0, 0, i));
	}

	/* Check x gradients */
	for (unsigned int i=0; i<3; ++i) {
		ASSERT_EQUALM("Wrong dx for entry " + std::to_string(i) , desired_dx(0, 0, i), dx(0, 0, i));
	}

}

