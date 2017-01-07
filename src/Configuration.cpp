/*
 * Configuration.cpp
 *
 *  Created on: Dec 16, 2016
 *      Author: ken
 */

#include "Configuration.hpp"

Configuration::Configuration(std::string configSrc) {
	mConfigSrc = configSrc;
	pugi::xml_parse_result result = mConfigDoc.load_file(mConfigSrc.c_str());
	std::cout << "Load result: " << result.description() << std::endl;
}

Configuration::Configuration(Configuration& other) {
	*this=other;
}

Configuration::~Configuration() {
}

const Configuration& Configuration::operator=(const Configuration& other) {
	mConfigSrc = other.mConfigSrc;
	pugi::xml_parse_result result = mConfigDoc.load_file(mConfigSrc.c_str());
	return *this;
}

unsigned int Configuration::getTrainingBatchSize() {
	return std::stoi(mConfigDoc.child("mochi-config").child("training-params").child("batch-size").child_value());
}

unsigned int Configuration::getNumEpochs() {
	return std::stoi(mConfigDoc.child("mochi-config").child("training-params").child("num-epochs").child_value());
}

std::vector<std::map<std::string, std::string>> Configuration::layerConfigs() {
	std::vector<std::map<std::string, std::string>> layerConfigs;
	pugi::xml_node layers = mConfigDoc.child("mochi-config").child("net").child("layers");
	for (pugi::xml_node_iterator layers_iter=layers.begin(); layers_iter != layers.end(); ++layers_iter) {
		std::map<std::string, std::string> layerConfig;
		for (pugi::xml_node_iterator layer_iter = layers_iter->begin(); layer_iter != layers_iter->end(); ++layer_iter) {
			layerConfig[layer_iter->name()] = layer_iter->child_value();
		}
		layerConfigs.push_back(layerConfig);
	}
	return layerConfigs;
}

std::map<std::string, std::string> Configuration::lossConfig() {
	std::map<std::string, std::string> lossConfig;
	pugi::xml_node loss = mConfigDoc.child("mochi-config").child("net").child("loss");
	for (pugi::xml_node_iterator config_iter = loss.begin(); config_iter != loss.end(); ++config_iter) {
		lossConfig[config_iter->name()] = config_iter->child_value();
	}
	return lossConfig;
}




