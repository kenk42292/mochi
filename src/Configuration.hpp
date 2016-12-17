/*
 * Configuration.hpp
 *
 *  Created on: Dec 16, 2016
 *      Author: ken
 */

#ifndef CONFIGURATION_HPP_
#define CONFIGURATION_HPP_

#include "pugixml.hpp"
#include <vector>
#include <map>
#include <iostream>

class Configuration {
private:
	pugi::xml_document configDoc;
public:
	Configuration(std::string configSrc);
	virtual ~Configuration();

	std::vector<std::map<std::string, std::string>> layerConfigs();
};

#endif /* CONFIGURATION_HPP_ */
