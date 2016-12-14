/*
 * NeuralNet.h
 *
 *  Created on: Dec 8, 2016
 *      Author: ken
 */

#ifndef NEURALNET_H_
#define NEURALNET_H_

#include "loss/Loss.h"
#include "layer/Layer.h"
#include "loss/Quadratic.h"

class NeuralNet {
private:
	std::vector<Layer> layers;
	Loss loss;
	unsigned int batchSize;
public:
	NeuralNet();
	virtual ~NeuralNet();

	arma::field<arma::Cube<double>> forwardPass(const arma::field<arma::Cube<double>>& inputs);
	arma::field<arma::Cube<double>> backwardPass(arma::field<arma::Cube<double>> deltas);
	void train(const arma::field<arma::Cube<double>>& inputs,
			const arma::field<arma::Cube<double>>& outputs);

};

#endif /* NEURALNET_H_ */
