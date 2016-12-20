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
		std::cout << "Creating Layer: " << layerType << std::endl;
		if (layerType.compare("vanillafeedforward")==0) {
			unsigned int dimIn = stoi(layerConfig["input-dim"]);
			unsigned int dimOut = stoi(layerConfig["output-dim"]);
			Optimizer* optimizer = createOptimizer(layerConfig);
			layer = new VanillaFeedForward(dimIn, dimOut, optimizer);
		} else if (layerType.compare("convolutional")==0) {
			//TODO: Implement this
		} else if (layerType.compare("sigmoid")==0) {
			layer = new Sigmoid();
		} else if (layerType.compare("softplus")==0) {
			layer = new Softplus();
		} else if (layerType.compare("softmax")==0) {
			layer = new Softmax();
		} else {
			//TODO: Perhaps have this throw an error...?
			std::cout << "Unimplemented Layer found" << std::endl;
		}
		layers.push_back(layer);
	}
	return layers;
}

Optimizer* LayerFactory::createOptimizer(std::map<std::string, std::string> layerConfig) {
	Optimizer* optimizer;
	std::string optimizerType = layerConfig["optimizer"];
	std::cout << "\t" << "Adding optimizer: " << optimizerType << std::endl;
	if (optimizerType.compare("gradientdescent")==0) {
		optimizer = new GradientDescent(stod(layerConfig["eta"]));
	} else if (optimizerType.compare("rmsprop")==0) {
		optimizer = new RMSProp(stod(layerConfig["eta"]));
	} else {
		std::cout << "Unimplemented Optimizer found" << std::endl;
	}
	return optimizer;
}





