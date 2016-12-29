/*
 * VanillaFeedForwardTest.hpp
 *
 *  Created on: Dec 28, 2016
 *      Author: ken
 */

#ifndef VANILLAFEEDFORWARDTEST_HPP_
#define VANILLAFEEDFORWARDTEST_HPP_

#include "cute.h"
#include "ide_listener.h"
#include "xml_listener.h"
#include "cute_runner.h"
#include <armadillo>
#include <iostream>

#include "../src/layer/VanillaFeedForward.hpp"
#include "../src/layer/optimizer/GradientDescent.hpp"

class VanillaFeedForwardTest {
public:
	VanillaFeedForwardTest();
	virtual ~VanillaFeedForwardTest();

	static void feedForwardTest1();
	static void feedForwardTest2();
	static void backPropTest1();
};

#endif /* VANILLAFEEDFORWARDTEST_HPP_ */
