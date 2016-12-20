/*
 * NeuralNet.h
 *
 *  Created on: Dec 8, 2016
 *      Author: ken
 */

#ifndef NEURALNET_HPP_
#define NEURALNET_HPP_

#include "layer/Layer.hpp"
#include "layer/Sigmoid.hpp"
#include "layer/VanillaFeedForward.hpp"
#include "loss/Loss.hpp"
#include "loss/Quadratic.hpp"
#include "loss/CrossEntropy.hpp"
#include "Configuration.hpp"
#include "layer/Sigmoid.hpp"
#include "layer/VanillaFeedForward.hpp"
#include "Utils.hpp"
#include <chrono>

class NeuralNet {
private:
	std::vector<Layer*> mLayers;
	Loss* mLoss;
public:
	NeuralNet(std::vector<Layer*> layers, Configuration conf);
	virtual ~NeuralNet();

	arma::field<arma::Cube<double>> forwardPass(const arma::field<arma::Cube<double>>& inputs);
	arma::field<arma::Cube<double>> backwardPass(arma::field<arma::Cube<double>> deltas);
	void train(arma::field<arma::Cube<double>>& inputs,
			arma::field<arma::Cube<double>>& outputs);

};

#endif /* NEURALNET_HPP_ */
