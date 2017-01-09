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
	std::string mConfigSrc;
	pugi::xml_document mConfigDoc;
public:
	Configuration(std::string configSrc);
	Configuration(Configuration& other);
	virtual ~Configuration();
	const Configuration& operator=(const Configuration& other);

	unsigned int batchSize();
	unsigned int numEpochs();
	std::vector<std::map<std::string, std::string>> layerConfigs();
	std::map<std::string, std::string> lossConfig() const;
};

#endif /* CONFIGURATION_HPP_ */
