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
	arma::field<arma::Cube<double>> test_images = loader.load_images(
			"./data/t10k-images-idx3-ubyte");
	arma::field<arma::Cube<double>> train_labels = loader.load_labels(
			"./data/train-labels-idx1-ubyte");
	arma::field<arma::Cube<double>> test_labels = loader.load_labels(
			"./data/t10k-labels-idx1-ubyte");

	/* For passing in to convolutional net - comment out as necessary */
	for (unsigned int i=0; i<train_images.size(); ++i) {
		train_images[i] = arma::Cube<double>((const double*) train_images[i].begin(), 28, 28, 1);
	}
	for (unsigned int i=0; i<test_images.size(); ++i) {
		test_images[i] = arma::Cube<double>((const double*) test_images[i].begin(), 28, 28, 1);
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
	std::cout << "fraction correct: " << nn.validate(test_images, test_labels, 5) << std::endl;



	return 0;
}
