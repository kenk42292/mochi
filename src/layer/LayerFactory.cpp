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
//	std::map<std::string, std::string> lossConfig = conf.lossConfig();



	unsigned int batchSize = conf.batchSize();

	Utils::printConfig(layerConfigs);
	std::vector<Layer*> layers;
	for (std::map<std::string, std::string> layerConfig : layerConfigs) {
		std::string layerType = layerConfig["type"];
		Layer* layer;
		double wdecay = 0.0;
		if (layerConfig.count("wdecay")==1) {
			wdecay = std::stod(layerConfig["wdecay"]);
		}
		std::cout << "Creating Layer: " << layerType << std::endl;
		if (layerType.compare("vanillafeedforward")==0) {
			unsigned int dimIn = stoi(layerConfig["input-dim"]);
			unsigned int dimOut = stoi(layerConfig["output-dim"]);
			Optimizer* optimizer = createOptimizer(layerConfig);
			layer = new VanillaFeedForward(batchSize, dimIn, dimOut, optimizer, wdecay);
		} else if (layerType.compare("convolutional")==0) {
			std::vector<unsigned int> inDim = Utils::parseDims(layerConfig["input-dim"]);
			unsigned int numPatterns = stoi(layerConfig["num-kernels"]);
			std::vector<unsigned int> kernelDim = Utils::parseDims(layerConfig["kernel-dim"]);
			std::vector<unsigned int> outDim = Utils::parseDims(layerConfig["output-dim"]);
			Optimizer* optimizer = createOptimizer(layerConfig);
			layer = new Convolutional(batchSize, inDim, numPatterns, kernelDim, outDim, optimizer, wdecay);
		} else if (layerType.compare("maxpool")==0) {
			std::vector<unsigned int> inDim = Utils::parseDims(layerConfig["input-dim"]);
			std::vector<unsigned int> fieldDim = Utils::parseDims(layerConfig["field-dim"]);
			std::vector<unsigned int> outDim = Utils::parseDims(layerConfig["output-dim"]);
			layer = new MaxPool(inDim, fieldDim, outDim);
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
	if (optimizerType.compare("sgd")==0) {
		optimizer = new SGD(stod(layerConfig["eta"]));
	} else if (optimizerType.compare("adagrad")==0) {
		optimizer = new Adagrad(stod(layerConfig["eta"]));
	} else if (optimizerType.compare("rmsprop")==0) {
		optimizer = new RMSProp(stod(layerConfig["eta"]), stod(layerConfig["gamma"]));
	}else {
		std::cout << "Unimplemented Optimizer found" << std::endl;
	}
	return optimizer;
}





