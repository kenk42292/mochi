/*
 * Configuration.cpp
 *
 *  Created on: Dec 16, 2016
 *      Author: ken
 */

#include "Configuration.hpp"

Configuration::Configuration(std::string configSrc) {
	pugi::xml_parse_result result = configDoc.load_file(configSrc.c_str());
	std::cout << "Load result: " << result.description() << std::endl;
}

Configuration::~Configuration() {
	// TODO Auto-generated destructor stub
}

std::vector<std::map<std::string, std::string>> Configuration::layerConfigs() {
	std::vector<std::map<std::string, std::string>> layerConfigs;
	pugi::xml_node layers = configDoc.child("mochi-config").child("net").child("layers");
	for (pugi::xml_node_iterator layers_iter=layers.begin(); layers_iter != layers.end(); ++layers_iter) {
		std::map<std::string, std::string> layerConfig;
		for (pugi::xml_node_iterator layer_iter = layers_iter->begin(); layer_iter != layers_iter->end(); ++layer_iter) {
			layerConfig[layer_iter->name()] = layer_iter->child_value();
		}
		layerConfigs.push_back(layerConfig);
	}
	return layerConfigs;
}


