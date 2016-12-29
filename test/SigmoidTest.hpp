/*
 * SigmoidTest.hpp
 *
 *  Created on: Dec 29, 2016
 *      Author: ken
 */

#ifndef SIGMOIDTEST_HPP_
#define SIGMOIDTEST_HPP_

#include "cute.h"
#include "ide_listener.h"
#include "xml_listener.h"
#include "cute_runner.h"
#include <armadillo>
#include <iostream>

#include "../src/layer/Sigmoid.hpp"

class SigmoidTest {
public:
	SigmoidTest();
	virtual ~SigmoidTest();

	static void feedForwardTest1();
	static void feedForwardTest2();
	static void backPropTest1();
	static void backPropTest2();
};

#endif /* SIGMOIDTEST_HPP_ */
