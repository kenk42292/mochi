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

class NeuralNet {
private:
	std::vector<Layer*> mLayers;
	Loss* loss;
public:
	NeuralNet(std::vector<Layer*> layers);
	virtual ~NeuralNet();

	arma::field<arma::Cube<double>> forwardPass(const arma::field<arma::Cube<double>>& inputs);
	arma::field<arma::Cube<double>> backwardPass(arma::field<arma::Cube<double>> deltas);
	void train(const arma::field<arma::Cube<double>>& inputs,
			const arma::field<arma::Cube<double>>& outputs);

};

#endif /* NEURALNET_HPP_ */
