/*
 * SigmoidTest.cpp
 *
 *  Created on: Dec 29, 2016
 *      Author: ken
 */

#include "SigmoidTest.hpp"

SigmoidTest::SigmoidTest() {}

SigmoidTest::~SigmoidTest() {}

void feedForwardTest1() {
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
	desired_ys(0) = arma::Cube<double>(1,1,3);
	// Desired output of x1: 0.73105858,  0.99330715,  0.95257413
	desired_ys(0)(0, 0, 0) = 0.73105858;
	desired_ys(0)(0, 0, 1) = 0.99330715;
	desired_ys(0)(0, 0, 2) = 0.95257413;
	// Desired output of x2: 0.98201379,  0.88079708,  0.99752738
	desired_ys(1) = arma::Cube<double>(1,1,3);
	desired_ys(1)(0, 0, 0) = 0.98201379;
	desired_ys(1)(0, 0, 1) = 0.88079708;
	desired_ys(1)(0, 0, 2) = 0.99752738;

	for (unsigned int i=0; i<2; ++i) {
		for (unsigned int j=0; j<3; ++j) {
			ASSERT_EQUAL_DELTAM("sigmoid outputs differ from expected", desired_ys(i)(j), ys(i)(j), 0.0001);
		}
	}
}

void feedForwardTest2() {

}

void backPropTest1() {

}

void backPropTest2() {

}




