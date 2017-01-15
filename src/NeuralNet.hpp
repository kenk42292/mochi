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
	NeuralNet(std::vector<Layer*> layers, Loss* loss);
	virtual ~NeuralNet();

	arma::field<arma::Cube<double>> forwardPass(
			const arma::field<arma::Cube<double>>& inputs);
	arma::field<arma::Cube<double>> backwardPass(
			arma::field<arma::Cube<double>> deltas);
	void train(const arma::field<arma::Cube<double>>& inputs,
			const arma::field<arma::Cube<double>>& outputs,
			unsigned int batchSize, unsigned int numEpochs, bool report);
	double validate(arma::field<arma::Cube<double>> inputs,
			arma::field<arma::Cube<double>> outputs, unsigned int k);
	unsigned int voteCategory(std::vector<arma::Cube<double>> probs);

};

#endif /* NEURALNET_HPP_ */
