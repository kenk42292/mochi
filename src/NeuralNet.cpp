/*
 * NeuralNet.cpp
 *
 *  Created on: Dec 8, 2016
 *      Author: ken
 */

#include "NeuralNet.h"
#include "layer/VanillaFeedForward.h"
#include "layer/Sigmoid.h"

NeuralNet::NeuralNet() {
	// TODO Auto-generated constructor stub
	batchSize = 1;
	VanillaFeedForward vff1(781, 300);
	Sigmoid s1;
	VanillaFeedForward vff2(300, 10);
	Sigmoid s2;
	layers = std::vector<Layer>(4);
	layers[0] = vff1;
	layers[1] = s1;
	layers[2] = vff2;
	layers[3] = s2;
	loss = Quadratic();
}

NeuralNet::~NeuralNet() {
	// TODO Auto-generated destructor stub
}

arma::field<arma::Cube<double>> NeuralNet::forwardPass(const arma::field<arma::Cube<double>>& inputs) {
	unsigned int numLayers = layers.size();
	arma::field<arma::Cube<double>> activations = inputs;
	for (unsigned int i=0; i<numLayers; ++i) {
		activations = layers[i].feedForward(activations);
	}
	return activations;
}

arma::field<arma::Cube<double>> NeuralNet::backwardPass(arma::field<arma::Cube<double>> deltas) {
	unsigned int numLayers = layers.size();
	for (int i=numLayers-1; i>=0; --i) {
		deltas = layers[i].backProp(deltas);
	}
	return deltas;
}

void NeuralNet::train(const arma::field<arma::Cube<double>>& inputs, const arma::field<arma::Cube<double>>& outputs) {

	unsigned int numLayers = 4;
	unsigned int batch_size = 10;

	for (unsigned int iter=0; iter<8; ++iter) {

		std::cout << "iter number: " << iter << std::endl;

		arma::field<arma::Cube<double>> activations = forwardPass(inputs);
		arma::field<arma::Cube<double>> deltas = loss.loss_prime(activations, outputs);
		backwardPass(deltas);

	}

	std::cout << "1000 iters completed" << std::endl;

}







