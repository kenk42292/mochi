/*
 * CrossEntropy.h
 *
 *  Created on: Dec 11, 2016
 *      Author: ken
 */

#ifndef LOSS_CROSSENTROPY_H_
#define LOSS_CROSSENTROPY_H_

#include "Loss.h"

class CrossEntropy: public Loss {
public:
	CrossEntropy();
	virtual ~CrossEntropy();
};

#endif /* LOSS_CROSSENTROPY_H_ */
