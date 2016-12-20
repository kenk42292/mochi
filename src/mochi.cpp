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
#include "layer/Layer.hpp"
#include "layer/LayerFactory.hpp"
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

	std::string confSrc = "config-sample.xml";
	Configuration conf(confSrc);
	LayerFactory layerFactory;

	std::vector<Layer*> layers = layerFactory.createLayers(conf);

	NeuralNet nn(layers, conf);
	unsigned int batchSize = conf.getTrainingBatchSize();
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	nn.train(train_images, train_labels, batchSize);
	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::seconds>(t2-t1).count();

	cout << "training complete" << endl;
	cout << "training duration: " << duration << endl;



	cout << "VALIDATING" << endl;
	double total_correct = 0;
	unsigned char prediction;
	arma::Cube<double> output;

	arma::field<arma::Cube<double>> predictions = nn.forwardPass(val_images);
	cout << "number of predictions: " << predictions.size() << endl;

	cout << "==================================" << endl;

	for (unsigned int i = 0; i < predictions.size(); ++i) {

		if (static_cast<int>(arma::vectorise(predictions[i]).index_max())
				== static_cast<int>(arma::vectorise(val_labels[i]).index_max())) {
			++total_correct;
		}
	}

	cout << "fraction correct: "
			<< static_cast<double>(total_correct / val_images.size()) << endl;


	return 0;
}
