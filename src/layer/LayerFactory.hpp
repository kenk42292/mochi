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

class LayerFactory {
public:
	LayerFactory();
	virtual ~LayerFactory();

	std::vector<Layer*> createLayers(Configuration conf);
};

#endif /* LAYER_LAYERFACTORY_HPP_ */
