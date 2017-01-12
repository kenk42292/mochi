//============================================================================
// Name        : mochi-init.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <armadillo>
#include <chrono>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "Configuration.hpp"
#include "layer/LayerFactory.hpp"
#include "loss/LossFactory.hpp"
#include "MNISTLoader.hpp"
#include "NeuralNet.hpp"
#include "Utils.hpp"

class LayerFactory;

using namespace std;

int main() {

	/** Load Data */
	MNIST_Loader loader;
	arma::field<arma::Cube<double>> train_images = loader.load_images(
			"./data/train-images-idx3-ubyte");
	arma::field<arma::Cube<double>> val_images = loader.load_images(
			"./data/t10k-images-idx3-ubyte");
	arma::field<arma::Cube<double>> train_labels = loader.load_labels(
			"./data/train-labels-idx1-ubyte");
	arma::field<arma::Cube<double>> val_labels = loader.load_labels(
			"./data/t10k-labels-idx1-ubyte");

	/* For passing in to convolutional net - comment out as necessary */
	for (unsigned int i=0; i<train_images.size(); ++i) {
		train_images[i] = arma::Cube<double>((const double*) train_images[i].begin(), 28, 28, 1);
	}
	for (unsigned int i=0; i<val_images.size(); ++i) {
		val_images[i] = arma::Cube<double>((const double*) val_images[i].begin(), 28, 28, 1);
	}


	std::string confSrc = "config-sample2.xml";
	Configuration conf(confSrc);
	LayerFactory layerFactory;
	LossFactory lossFactory;

	std::vector<Layer*> layers = layerFactory.createLayers(conf);
	Loss* loss = lossFactory.createLoss(conf);

	NeuralNet nn(layers, loss);
	unsigned int batchSize = conf.batchSize();
	unsigned int numEpochs = conf.numEpochs();
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	nn.train(train_images, train_labels, batchSize, numEpochs, true);
	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::seconds>(t2-t1).count();

	cout << "training complete" << endl;
	cout << "training duration: " << duration << endl;



	cout << "validating..." << endl;
	double total_correct = 0;
	unsigned char prediction;
	arma::Cube<double> output;

	arma::field<arma::Cube<double>> predictions1 = nn.forwardPass(val_images);
	arma::field<arma::Cube<double>> predictions2 = nn.forwardPass(val_images);
	arma::field<arma::Cube<double>> predictions3 = nn.forwardPass(val_images);

	for (unsigned int i = 0; i < predictions1.size(); ++i) {
		unsigned int actual = static_cast<unsigned int>(arma::vectorise(val_labels[i]).index_max());

		unsigned int p1 = static_cast<unsigned int>(arma::vectorise(predictions1[i]).index_max());
		double p1p = arma::vectorise(predictions1[i]).max();

		unsigned int p2 = static_cast<unsigned int>(arma::vectorise(predictions2[i]).index_max());
		double p2p = arma::vectorise(predictions2[i]).max();

		unsigned int p3 = static_cast<unsigned int>(arma::vectorise(predictions3[i]).index_max());
		double p3p = arma::vectorise(predictions3[i]).max();

		unsigned int guess = p1;
		if (p2==p3) {
			guess = p2;
		}

		if (p1!=p2 && p2!=p3 && p3!=p1) {
			guess = p1;
			if (p2p>p1p && p2p>p3p) {
				guess = p2;
			}
			if (p3p>p2p && p3p>p1p) {
				guess = p3;
			}
		}

//		std::cout << "----------------------------------------" << std::endl;
//		std::cout << "guesss: " << p1 << ", " << p2 << ", " << p3 << std::endl;
//		std::cout << "probs: " << p1p << ", " << p2p << ", " << p3p << std::endl;
//		std::cout << "final guess: " << guess << std::endl;
//		std::cout << "actual: " << actual << std::endl;

		if (guess == actual) {
			++total_correct;
		}
	}
	cout << "fraction correct: "
			<< static_cast<double>(total_correct / val_images.size()) << endl;

	return 0;
}
