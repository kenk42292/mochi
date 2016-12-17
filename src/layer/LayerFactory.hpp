/*
 * LayerFactory.hpp
 *
 *  Created on: Dec 16, 2016
 *      Author: ken
 */

#ifndef LAYER_LAYERFACTORY_HPP_
#define LAYER_LAYERFACTORY_HPP_

#include <vector>
#include "Layer.hpp"
#include "../Configuration.hpp"
#include "../Utils.hpp"
#include "../layer/VanillaFeedForward.hpp"
//#include "../layer/Convolutional.hpp"
#include "../layer/Sigmoid.hpp"
#include "../layer/Softplus.hpp"
#include "../layer/Softmax.hpp"

class LayerFactory {
public:
	LayerFactory();
	virtual ~LayerFactory();

	std::vector<Layer*> createLayers(Configuration conf);
};

#endif /* LAYER_LAYERFACTORY_HPP_ */
