/*
 * VanillaFeedForwardTest.cpp
 *
 *  Created on: Dec 28, 2016
 *      Author: ken
 */

#include "VanillaFeedForwardTest.hpp"

void VanillaFeedForwardTest::feedForwardTest1() {
	Optimizer* optimizer = new GradientDescent(0.1);
	VanillaFeedForward vff(3, 2, optimizer);
	arma::Cube<double> w(2, 3, 1);
	for (unsigned int i=0; i<2; ++i) {
		for (unsigned int j=0; j<3; ++j) {
			w(i, j, 0) = 3*i+j;
		}
	}
	arma::Cube<double> b(1, 1, 2);
	b(0, 0, 0) = 3;
	b(0, 0, 1) = 5;
	vff.mW = w;
	vff.mB = b;
	arma::Cube<double> input(1, 1, 3);
	input(0, 0, 0) = 2;
	input(0, 0, 1) = 4;
	input(0, 0, 2) = 6;
	arma::Cube<double> desiredOutput(1, 1, 2);
	desiredOutput(0, 0, 0) = 19;
	desiredOutput(0, 0, 1) = 57;
	arma::Cube<double> output = vff.feedForward(input);
	ASSERT_EQUAL(2, output.n_slices);
	ASSERT_EQUAL(1, output.n_rows);
	ASSERT_EQUAL(1, output.n_cols);
	for (unsigned int i=0; i<2; ++i) {
		ASSERT_EQUALM("Desired output for vff feedforward mismatch", desiredOutput(0, 0, i), output(0, 0, i));
	}
}

void VanillaFeedForwardTest::feedForwardTest2() {
	Optimizer* optimizer = new GradientDescent(0.1);
	VanillaFeedForward vff(8, 4, optimizer);
	arma::Cube<double> w(4, 8, 2);
	for (unsigned int i=0; i<4; ++i) {
		for (unsigned int j=0; j<8; ++j) {
			w(i, j, 0) = 8*i+j;
		}
	}
	arma::Cube<double> b(1, 1, 4);
	b(0, 0, 0) = 3;
	b(0, 0, 1) = 5;
	b(0, 0, 2) = 1;
	b(0, 0, 3) = 9;
	vff.mW = w;
	vff.mB = b;
	arma::Cube<double> input(2, 2, 2);
	for (unsigned int k=0; k<2; ++k) {
		for (unsigned int i=0; i<2; ++i) {
			for (unsigned int j=0; j<2; ++j) {
				input(i, j, k) = 4*k+2*i+j;
			}
		}
	}
	arma::Cube<double> desiredOutput(1, 1, 4);
	desiredOutput(0, 0, 0) = 141;
	desiredOutput(0, 0, 1) = 367;
	desiredOutput(0, 0, 2) = 587;
	desiredOutput(0, 0, 3) = 819;
	arma::Cube<double> output = vff.feedForward(input);
	ASSERT_EQUAL(4, output.n_slices);
	ASSERT_EQUAL(1, output.n_rows);
	ASSERT_EQUAL(1, output.n_cols);
	for (unsigned int i=0; i<4; ++i) {
		ASSERT_EQUALM("Desired output for vff feedforward mismatch", output(0, 0, i), desiredOutput(0, 0, i));
	}
}


void VanillaFeedForwardTest::backPropTest1() {
	Optimizer* optimizer = new GradientDescent(0.1);
	VanillaFeedForward vff(8, 4, optimizer);
	arma::Cube<double> w(4, 8, 2);
	for (unsigned int i=0; i<4; ++i) {
		for (unsigned int j=0; j<8; ++j) {
			w(i, j, 0) = 8*i+j;
		}
	}
	arma::Cube<double> b(1, 1, 4);
	b(0, 0, 0) = 3;
	b(0, 0, 1) = 5;
	b(0, 0, 2) = 1;
	b(0, 0, 3) = 9;
	vff.mW = w;
	vff.mB = b;
	arma::Cube<double> input(2, 2, 2);
	for (unsigned int k=0; k<2; ++k) {
		for (unsigned int i=0; i<2; ++i) {
			for (unsigned int j=0; j<2; ++j) {
				input(i, j, k) = 4*k+2*i+j;
			}
		}
	}
	arma::Cube<double> desiredOutput(1, 1, 4);
	desiredOutput(0, 0, 0) = 141;
	desiredOutput(0, 0, 1) = 367;
	desiredOutput(0, 0, 2) = 587;
	desiredOutput(0, 0, 3) = 819;
	arma::Cube<double> output = vff.feedForward(input);
	ASSERT_EQUAL(4, output.n_slices);
	ASSERT_EQUAL(1, output.n_rows);
	ASSERT_EQUAL(1, output.n_cols);
	for (unsigned int i=0; i<4; ++i) {
		ASSERT_EQUALM("Desired output for vff feedforward mismatch", output(0, 0, i), desiredOutput(0, 0, i));
	}
}


cute::suite make_suite(){
	cute::suite s;
	s.push_back(CUTE(VanillaFeedForwardTest::feedForwardTest2));
	s.push_back(CUTE(VanillaFeedForwardTest::feedForwardTest1));
	s.push_back(CUTE(VanillaFeedForwardTest::backPropTest1));
	return s;
}
