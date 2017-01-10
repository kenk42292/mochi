/*
 * ConvolutionalTest.hpp
 *
 *  Created on: Jan 3, 2017
 *      Author: ken
 */

#ifndef CONVOLUTIONALTEST_HPP_
#define CONVOLUTIONALTEST_HPP_

#include "cute.h"
#include "ide_listener.h"
#include "xml_listener.h"
#include "cute_runner.h"
#include <armadillo>
#include <iostream>

#include "../src/layer/Convolutional.hpp"
#include "../src/layer/optimizer/SGD.hpp"

class ConvolutionalTest {
public:
	ConvolutionalTest();
	virtual ~ConvolutionalTest();

	static arma::field<arma::Cube<double>> mockWeights();
	static arma::Cube<double> mockBias();
	static void feedForwardTest1();
	static void feedForwardTest2();
	static void feedForwardTest3();
	static void backPropTest1();
	static void backPropTest2();
	static void backPropTest3();
};

#endif /* CONVOLUTIONALTEST_HPP_ */
