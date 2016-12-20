/*
 * RMSProp.hpp
 *
 *  Created on: Dec 19, 2016
 *      Author: ken
 */

#ifndef LAYER_OPTIMIZER_RMSPROP_HPP_
#define LAYER_OPTIMIZER_RMSPROP_HPP_

#include "Optimizer.hpp"

class RMSProp: public Optimizer {
private:
	double mEta;
public:
	RMSProp(double eta);
	virtual ~RMSProp();
};

#endif /* LAYER_OPTIMIZER_RMSPROP_HPP_ */
