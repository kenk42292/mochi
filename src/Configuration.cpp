/*
 * Configuration.cpp
 *
 *  Created on: Dec 16, 2016
 *      Author: ken
 */

#include "Configuration.hpp"

Configuration::Configuration(const char* configSrc) {
	pugi::xml_parse_result result = configDoc.load_file(configSrc);
//	std::cout << "Load result: " << result.description() << std::endl;
}

Configuration::~Configuration() {
	// TODO Auto-generated destructor stub
}


