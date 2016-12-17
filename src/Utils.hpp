/*
 * Utils.h
 *
 *  Created on: Dec 16, 2016
 *      Author: ken
 */

#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <cstdlib>
#include <armadillo>
#include <vector>
#include <map>

class Utils {
public:
	Utils();
	virtual ~Utils();

	static void shuffle(arma::field<arma::Cube<double>>& inputs, arma::field<arma::Cube<double>>& outputs);
	static void printConfig(std::vector<std::map<std::string, std::string>> layersConfig);
};

#endif /* UTILS_HPP_ */
