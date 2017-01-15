/*
 * NeuralNet.cpp
 *
 *  Created on: Dec 8, 2016
 *      Author: ken
 */

#include "NeuralNet.hpp"

NeuralNet::NeuralNet(std::vector<Layer*> layers, Loss* loss) :
		mLayers(layers), mLoss(loss) {
}

NeuralNet::~NeuralNet() {
	for (Layer* lp : mLayers) {
		delete lp;
	}
	delete mLoss;
}

arma::field<arma::Cube<double>> NeuralNet::forwardPass(
		const arma::field<arma::Cube<double>>& inputs) {
//	std::cout << "NEURALNET FORWARDPASS" << std::endl;
	unsigned int numLayers = mLayers.size();
	arma::field<arma::Cube<double>> activations = inputs;
	for (unsigned int i = 0; i < numLayers; ++i) {
		activations = mLayers[i]->feedForward(activations);
	}
	return activations;
}

arma::field<arma::Cube<double>> NeuralNet::backwardPass(
		arma::field<arma::Cube<double>> deltas) {
//	std::cout << "NEURALNET BACKWARDPASS" << std::endl;
	unsigned int numLayers = mLayers.size();
	for (int i = numLayers - 1; i >= 0; --i) { //TODO: Remember to change this back...
		deltas = mLayers[i]->backProp(deltas);
	}
	return deltas;
}

void NeuralNet::train(const arma::field<arma::Cube<double>>& inputs,
		const arma::field<arma::Cube<double>>& outputs, unsigned int batchSize,
		unsigned int numEpochs, bool log) {

	unsigned int vset_size = inputs.size() / 10;
	unsigned int tset_size = inputs.size() - vset_size;

	arma::field<arma::Cube<double>> t_inputs = inputs.rows(0, tset_size - 1);
	arma::field<arma::Cube<double>> t_outputs = outputs.rows(0, tset_size - 1);
	arma::field<arma::Cube<double>> v_inputs = inputs.rows(tset_size,
			inputs.size() - 1);
	arma::field<arma::Cube<double>> v_outputs = outputs.rows(tset_size,
			outputs.size() - 1);

	for (unsigned int ep = 0; ep < numEpochs; ++ep) {
		std::cout << "epoch: " << ep << std::endl;
		std::chrono::high_resolution_clock::time_point t0 =
				std::chrono::high_resolution_clock::now();
		double epochLoss = 0;
		Utils::shuffle(t_inputs, t_outputs);
		for (unsigned int p = 0; p < t_inputs.size() - batchSize; p +=
				batchSize) {
//		for (unsigned int p=0; p<600; p+=batchSize) {
			arma::field<arma::Cube<double>> activations = forwardPass(
					t_inputs.rows(p, p + batchSize - 1));
			arma::field<arma::Cube<double>> deltas = mLoss->loss_prime(
					activations, t_outputs.rows(p, p + batchSize - 1));
			backwardPass(deltas);
			if (log) {
				epochLoss += mLoss->loss(activations,
						t_outputs.rows(p, p + batchSize - 1));
			}
		}
		if (log) {
			std::chrono::high_resolution_clock::time_point t =
					std::chrono::high_resolution_clock::now();
			std::cout << "\tEpoch Duration: "
					<< std::chrono::duration_cast<std::chrono::seconds>(t - t0).count()
					<< std::endl;
			std::cout << "\tTraining Loss: " << epochLoss << std::endl;
			arma::field<arma::Cube<double>> v_activations = forwardPass(v_inputs);
			double v_loss = mLoss->loss(v_activations, v_outputs);
			std::cout << "\tValidation Loss: " << v_loss << std::endl;
			std::cout << "\tValidation Error Rate: "
					<< validate(v_inputs, v_outputs, 5) << std::endl;
		}

	}

}

double NeuralNet::validate(arma::field<arma::Cube<double>> inputs,
		arma::field<arma::Cube<double>> outputs, unsigned int k=1) {

	double total_correct = 0;

	std::vector<arma::field<arma::Cube<double>>> kpredictions;
	// Produce complete answers, k times
	for (unsigned int i = 0; i < k; ++i) {
		kpredictions.push_back(forwardPass(inputs));
	}

	for (unsigned int i = 0; i < inputs.size(); ++i) {
		unsigned int actual = arma::vectorise(outputs[i]).index_max();

		std::vector<arma::Cube<double>> sample_dist;
		for (unsigned int j = 0; j < k; ++j) {
			sample_dist.push_back(kpredictions[j][i]);
		}

		unsigned int guess = voteCategory(sample_dist);

		if (guess == actual) {
			++total_correct;
		}
	}

	return static_cast<double>(total_correct / inputs.size());
}


/** Returns index of most likely element, given a series of probability distributions.
 * The most likely element is voted on by majority votes. Ties are broken by selecting first element seen. */
unsigned int NeuralNet::voteCategory(std::vector<arma::Cube<double>> probs) {
	arma::Cube<unsigned int> counts(1, 1, probs[0].size(), arma::fill::zeros);
	for (unsigned int i=0; i<probs.size(); ++i) {
		unsigned int ml_i = arma::vectorise(probs[i]).index_max();
		counts[ml_i]+=1;
	}
	unsigned int maj_count = 0;
	unsigned int maj_ind = 0;
	for (unsigned int i=0; i<counts.size(); ++i) {
		if (counts[i]>maj_count) {
			maj_count = counts[i];
			maj_ind = i;
		}
	}
	return maj_ind;
}
