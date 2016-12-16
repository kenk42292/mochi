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
	VanillaFeedForward* vff1;
	Sigmoid* s1;
	VanillaFeedForward* vff2;
	Sigmoid* s2;
	std::vector<Layer*> layers;
	Loss* loss;
	unsigned int batchSize;
public:
	NeuralNet();
	virtual ~NeuralNet();

	arma::field<arma::Cube<double>> forwardPass(const arma::field<arma::Cube<double>>& inputs);
	arma::field<arma::Cube<double>> backwardPass(arma::field<arma::Cube<double>> deltas);
	void train(const arma::field<arma::Cube<double>>& inputs,
			const arma::field<arma::Cube<double>>& outputs);

};

#endif /* NEURALNET_HPP_ */
