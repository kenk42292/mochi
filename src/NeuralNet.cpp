/*
 * NeuralNet.cpp
 *
 *  Created on: Dec 8, 2016
 *      Author: ken
 */

#include "NeuralNet.h"

NeuralNet::NeuralNet() {
	// TODO Auto-generated constructor stub

}

NeuralNet::~NeuralNet() {
	// TODO Auto-generated destructor stub
}

void NeuralNet::train(const std::vector<arma::Col<double>>& inputs, const std::vector<arma::Col<double>>& outputs) {
	std::vector<arma::Col<double>>& activations = inputs;
	for (Layer layer: layers) {
		activations = layer.feedForward(activations);
	}
	//TODO: Depending on configuration, best way to get delta is...?
	std::vector<arma::Col<double>> deltas = loss.
}

void train(const std::vector<arma::Mat<double>>& inputs, const std::vector<arma::Col<double>>& outputs) {

}

void train(const std::vector<arma::Cube<double>>& inputs, const std::vector<arma::Col<double>>& outputs) {

}


