/*
 * CrossEntropy.h
 *
 *  Created on: Dec 11, 2016
 *      Author: ken
 */

#ifndef LOSS_CROSSENTROPY_HPP_
#define LOSS_CROSSENTROPY_HPP_

#include "Loss.hpp"

class CrossEntropy: public Loss {
public:
	CrossEntropy();
	virtual ~CrossEntropy();
};

#endif /* LOSS_CROSSENTROPY_HPP_ */
