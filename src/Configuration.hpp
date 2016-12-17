/*
 * Configuration.hpp
 *
 *  Created on: Dec 16, 2016
 *      Author: ken
 */

#ifndef CONFIGURATION_HPP_
#define CONFIGURATION_HPP_

#include "pugixml.hpp"

class Configuration {
private:
	pugi::xml_document configDoc;
public:
	Configuration(const char* configSrc);
	virtual ~Configuration();
};

#endif /* CONFIGURATION_HPP_ */
