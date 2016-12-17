/*
 * NeuralNet.cpp
 *
 *  Created on: Dec 8, 2016
 *      Author: ken
 */

#include "NeuralNet.hpp"

#include "layer/Sigmoid.hpp"
#include "layer/VanillaFeedForward.hpp"

NeuralNet::NeuralNet(std::vector<Layer*> layers) {
	// TODO Auto-generated constructor stub
	mLayers = layers;
	loss = new Quadratic();
}

NeuralNet::~NeuralNet() {
	// TODO Auto-generated destructor stub
//	delete vff1;
//	delete s1;
//	delete vff2;
//	delete s2;
	delete loss;
}

arma::field<arma::Cube<double>> NeuralNet::forwardPass(const arma::field<arma::Cube<double>>& inputs) {
	unsigned int numLayers = mLayers.size();
	arma::field<arma::Cube<double>> activations = inputs;
	for (unsigned int i=0; i<numLayers; ++i) {
		activations = mLayers[i]->feedForward(activations);
	}
	return activations;
}

arma::field<arma::Cube<double>> NeuralNet::backwardPass(arma::field<arma::Cube<double>> deltas) {
	unsigned int numLayers = mLayers.size();
	for (int i=numLayers-1; i>=0; --i) {
		deltas = mLayers[i]->backProp(deltas);
	}
	return deltas;
}

void NeuralNet::train(const arma::field<arma::Cube<double>>& inputs, const arma::field<arma::Cube<double>>& outputs) {

	unsigned int numLayers = 4;
	unsigned int numEpochs = 5;
	unsigned int batchSize = 100;




	for (unsigned int ep=0; ep<numEpochs; ++ep) {

		std::cout << "epoch: " << ep << std::endl;

		for (unsigned int p=0; p<inputs.size()-batchSize; p+=batchSize) {


			arma::field<arma::Cube<double>> activations = forwardPass(inputs.rows(p, p+batchSize-1));

			arma::field<arma::Cube<double>> deltas = loss->loss_prime(activations, outputs.rows(p, p+batchSize));

			backwardPass(deltas);

		}
	}

}







