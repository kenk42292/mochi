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
	if (lossConfig.compare("quadratic") == 0) {
		std::cout << "Setting to Quadratic Loss function" << std::endl;
		mLoss = new Quadratic();
	} else if (lossConfig.compare("crossentropy") == 0) {
		std::cout << "Setting to Cross Entropy Loss function" << std::endl;
		mLoss = new CrossEntropy();
	} else {
		mLoss = new CrossEntropy();
		std::cout << "No configured loss fxn. Setting to cross entropy"
				<< std::endl;
	}
}

NeuralNet::~NeuralNet() {
	for (Layer* lp : mLayers) {
		delete lp;
	}
	delete mLoss;
}

arma::field<arma::Cube<double>> NeuralNet::forwardPass(
		const arma::field<arma::Cube<double>>& inputs) {
	unsigned int numLayers = mLayers.size();
	arma::field<arma::Cube<double>> activations = inputs;
	for (unsigned int i = 0; i < numLayers; ++i) {
		activations = mLayers[i]->feedForward(activations);
	}
	return activations;
}

arma::field<arma::Cube<double>> NeuralNet::backwardPass(
		arma::field<arma::Cube<double>> deltas) {
	unsigned int numLayers = mLayers.size();
	for (int i = numLayers - 1; i >= 0; --i) { //TODO: Remember to change this back...
		deltas = mLayers[i]->backProp(deltas);
	}
	return deltas;
}

void NeuralNet::train(arma::field<arma::Cube<double>>& inputs,
		arma::field<arma::Cube<double>>& outputs, unsigned int batchSize,
		unsigned int numEpochs, bool report) {

//	long ff(0);
//	long l(0);
//	long bp(0);

	for (unsigned int ep = 0; ep < numEpochs; ++ep) {
		std::cout << "epoch: " << ep << std::endl;
		Utils::shuffle(inputs, outputs);
		for (unsigned int p = 0; p < inputs.size() - batchSize; p +=
				batchSize) {
//			std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
			arma::field<arma::Cube<double>> activations = forwardPass(
					inputs.rows(p, p + batchSize - 1));
//			std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
			arma::field<arma::Cube<double>> deltas = mLoss->loss_prime(
					activations, outputs.rows(p, p + batchSize - 1));
//			std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();
			backwardPass(deltas);
//			std::chrono::high_resolution_clock::time_point t4 = std::chrono::high_resolution_clock::now();

//			ff += std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
//			l += std::chrono::duration_cast<std::chrono::microseconds>(t3-t2).count();
//			bp += std::chrono::duration_cast<std::chrono::microseconds>(t4-t3).count();
		}
		if (report) {
			arma::field<arma::Cube<double>> activations = forwardPass(inputs.rows(0, batchSize - 1));
			std::cout << "Single Batch Loss: " << mLoss->loss(activations, outputs.rows(0, batchSize-1))<<std::endl;
		}

	}
//
//	std::cout << "feedforward time: " << ff/1000000 << std::endl;
//	std::cout << "loss time: " << l/1000000 << std::endl;
//	std::cout << "bp time: " << bp/1000000 << std::endl;

}

