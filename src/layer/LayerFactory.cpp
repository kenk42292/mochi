/*
 * LayerFactory.cpp
 *
 *  Created on: Dec 16, 2016
 *      Author: ken
 */

#include "LayerFactory.hpp"

LayerFactory::LayerFactory() {}

LayerFactory::~LayerFactory() {}

std::vector<Layer*> LayerFactory::createLayers(Configuration conf) {
	std::vector<std::map<std::string, std::string>> layerConfigs = conf.layerConfigs();
	Utils::printConfig(layerConfigs);
	std::vector<Layer*> layers;
	for (std::map<std::string, std::string> layerConfig : layerConfigs) {
		std::string layerType = layerConfig["type"];
		Layer* layer;
		if (layerType.compare("vanillafeedforward")==0) {
			std::cout << "Creating Vanilla Feedforward Layer..." << std::endl;
			unsigned int dimIn = stoi(layerConfig["input-dim"]);
			unsigned int dimOut = stoi(layerConfig["output-dim"]);
			layer = new VanillaFeedForward(dimIn, dimOut);
		} else if (layerType.compare("convolutional")==0) {
			//TODO: Implement this
		} else if (layerType.compare("sigmoid")==0) {
			std::cout << "Creating Sigmoid Layer..." << std::endl;
			layer = new Sigmoid();
		} else if (layerType.compare("softplus")==0) {
			//TODO: Implement this
		} else if (layerType.compare("softmax")==0) {
			//TODO: Implement this
		} else {
			//TODO: Perhaps have this throw an error...?
			std::cout << "Unimplemented Layer found" << std::endl;
		}
		layers.push_back(layer);
	}
	return layers;
}
