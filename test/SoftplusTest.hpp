/*
 * SoftplusTest.h
 *
 *  Created on: Dec 30, 2016
 *      Author: ken
 */

#ifndef SOFTPLUSTEST_HPP_
#define SOFTPLUSTEST_HPP_

#include "cute.h"
#include "ide_listener.h"
#include "xml_listener.h"
#include "cute_runner.h"
#include <armadillo>
#include <iostream>

#include "../src/layer/Softplus.hpp"

class SoftplusTest {
public:
	SoftplusTest();
	virtual ~SoftplusTest();

	static void feedForwardTest1();
	static void feedForwardTest2();
	static void backPropTest1();
	static void backPropTest2();
};

#endif /* SOFTPLUSTEST_HPP_ */
