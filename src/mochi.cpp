//============================================================================
// Name        : mochi-init.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <armadillo>
#include <vector>

#include "layer/Sigmoid.hpp"
#include "layer/VanillaFeedForward.hpp"
#include "MNISTLoader.hpp"
#include "NeuralNet.hpp"
using namespace std;

int main() {
	cout << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!

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

//
//	const char* confSrc = "config-sample.xml";
//	Configuration conf(confSrc);

	VanillaFeedForward vff1(784, 300);
	Sigmoid s1;
	VanillaFeedForward vff2(300, 10);
	Sigmoid s2;

	std::vector<Layer*> layers(4);
	layers[0] = &vff1;
	layers[1] = &s1;
	layers[2] = &vff2;
	layers[3] = &s2;

	NeuralNet nn(layers);
	nn.train(train_images, train_labels);

	cout << "training complete" << endl;



	cout << "VALIDATING" << endl;
	double total_correct = 0;
	unsigned char prediction;
	arma::Cube<double> output;

	arma::field<arma::Cube<double>> predictions = nn.forwardPass(val_images);
	cout << "number of predictions: " << predictions.size() << endl;

	cout << "==================================" << endl;

	for (unsigned int i = 0; i < predictions.size(); ++i) {

//		cout << "======================================" << endl;
//		cout << "prediction: " << static_cast<int>(arma::vectorise(predictions[i]).index_max()) << endl;
//		cout << "actual: " << static_cast<int>(arma::vectorise(val_labels[i]).index_max()) << endl;

		if (static_cast<int>(arma::vectorise(predictions[i]).index_max())
				== static_cast<int>(arma::vectorise(val_labels[i]).index_max())) {
			++total_correct;
		}
	}

	cout << "fraction correct: "
			<< static_cast<double>(total_correct / val_images.size()) << endl;


//	std::cout << f.rows(0, 2) << std::endl;



	return 0;
}
