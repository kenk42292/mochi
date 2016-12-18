/*
 * NeuralNet.cpp
 *
 *  Created on: Dec 8, 2016
 *      Author: ken
 */

#include "NeuralNet.hpp"

NeuralNet::NeuralNet(std::vector<Layer*> layers, Configuration conf) {
	mLayers = layers;
	std::string lossConfig = conf.lossConfig();
	if (lossConfig.compare("quadratic")==0) {
		std::cout << "Setting to Quadratic Loss function" << std::endl;
		mLoss = new Quadratic();
	} else if (lossConfig.compare("crossentropy")==0) {
		std::cout << "Setting to Cross Entropy Loss function" << std::endl;
		mLoss = new CrossEntropy();
	} else {
		mLoss = new CrossEntropy();
		std::cout << "No configured loss fxn. Setting to cross entropy" << std::endl;
	}
}

NeuralNet::~NeuralNet() {
	for (Layer* lp : mLayers) {
		delete lp;
	}
	delete mLoss;
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

void NeuralNet::train(arma::field<arma::Cube<double>>& inputs, arma::field<arma::Cube<double>>& outputs) {

	unsigned int numLayers = 4;
	unsigned int numEpochs = 2;
	unsigned int batchSize = 5;

	for (unsigned int ep=0; ep<numEpochs; ++ep) {
		std::cout << "epoch: " << ep << std::endl;
		Utils::shuffle(inputs, outputs);
		for (unsigned int p=0; p<inputs.size()-batchSize; p+=batchSize) {
			arma::field<arma::Cube<double>> activations = forwardPass(inputs.rows(p, p+batchSize-1));
			arma::field<arma::Cube<double>> deltas = mLoss->loss_prime(activations, outputs.rows(p, p+batchSize-1));
			backwardPass(deltas);
		}
	}

}







