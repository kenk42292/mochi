/*
 * Utils.cpp
 *
 *  Created on: Dec 16, 2016
 *      Author: ken
 */

#include "Utils.hpp"

Utils::Utils() {}

Utils::~Utils() {}

std::vector<unsigned int> Utils::parseDims(std::string s) {
    std::vector<unsigned int> result;
    std::stringstream ss(s);
    unsigned int dim;
    while (ss >> dim) {
        while (ss.peek()==',' || ss.peek()==' ') {
            ss.ignore();
        }
        result.push_back(dim);
    }
    return result;
}

arma::field<arma::Cube<double>> Utils::flipCubes(const arma::field<arma::Cube<double>>& f) {
	arma::field<arma::Cube<double>> flipped(f.size());
	for (unsigned int i=0; i<f.size(); ++i) {
		arma::Cube<double> flippedCube(f[i].n_rows, f[i].n_cols, f[i].n_slices);
		for (unsigned int j=0; j<f[i].n_slices; ++j) {
			flippedCube.slice(j) = arma::fliplr(arma::flipud(f[i].slice(j)));
		}
		flipped[i] = flippedCube;
	}
	return flipped;
}

void Utils::shuffle(arma::field<arma::Cube<double>>& inputs, arma::field<arma::Cube<double>>& outputs) {
	unsigned int swapIndex;
	unsigned int n = inputs.size();
	for (unsigned int i=0; i<n; ++i) {
		swapIndex = i + (rand()%(n-i));
		arma::Cube<double> tempIn = inputs[i];
		arma::Cube<double> tempOut = outputs[i];
		inputs[i] = inputs[swapIndex];
		outputs[i] = outputs[swapIndex];
		inputs[swapIndex] = tempIn;
		outputs[swapIndex] = tempOut;
	}
}

void Utils::printConfig(std::vector<std::map<std::string, std::string>> layersConfig) {
	for (auto m : layersConfig) {
		std::cout << "LAYER---------------------------------------" << std::endl;
		for (auto const& pair : m) {
			std::cout << "\t\t" << pair.first << " : " << pair.second << std::endl;
		}
	}
}






