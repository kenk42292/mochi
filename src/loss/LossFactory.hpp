/*
 * LossFactory.hpp
 *
 *  Created on: Jan 7, 2017
 *      Author: ken
 */

#ifndef LOSS_LOSSFACTORY_HPP_
#define LOSS_LOSSFACTORY_HPP_

#include <string>
#include <iostream>

#include "../Configuration.hpp"
#include "Loss.hpp"
#include "Quadratic.hpp"
#include "CrossEntropy.hpp"

class LossFactory {
public:
	LossFactory();
	virtual ~LossFactory();
	Loss* createLoss(Configuration conf);
};

#endif /* LOSS_LOSSFACTORY_HPP_ */
