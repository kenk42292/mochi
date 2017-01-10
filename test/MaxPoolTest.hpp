/*
 * MaxPoolTest.hpp
 *
 *  Created on: Jan 2, 2017
 *      Author: ken
 */

#ifndef MAXPOOLTEST_HPP_
#define MAXPOOLTEST_HPP_

#include "cute.h"
#include "ide_listener.h"
#include "xml_listener.h"
#include "cute_runner.h"
#include <armadillo>
#include <iostream>

#include "../src/layer/MaxPool.hpp"

class MaxPoolTest {
public:
	MaxPoolTest();
	virtual ~MaxPoolTest();

	static void feedForwardTest1();
	static void feedForwardTest2();
	static void backPropTest1();
	static void backPropTest2();
};

#endif /* MAXPOOLTEST_HPP_ */
