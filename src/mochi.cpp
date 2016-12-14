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

#include "layer/Sigmoid.h"
#include "layer/VanillaFeedForward.h"
#include "MNISTLoader.h"
#include "NeuralNet.h"
using namespace std;

int main() {
	cout << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!

	/** Load Data */
	/*
	MNIST_Loader loader;
	arma::field<arma::Cube<double>> train_images = loader.load_images(
			"./data/train-images-idx3-ubyte");
	arma::field<arma::Cube<double>> val_images = loader.load_images(
			"./data/t10k-images-idx3-ubyte");
	arma::field<arma::Cube<double>> train_labels = loader.load_labels(
			"./data/train-labels-idx1-ubyte");
	arma::field<arma::Cube<double>> val_labels = loader.load_labels(
			"./data/t10k-labels-idx1-ubyte");

	cout << train_labels[0] << endl;


	NeuralNet nn;
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

		cout << "======================================" << endl;
		cout << "prediction: " << static_cast<int>(arma::vectorise(predictions[i]).index_max()) << endl;
		cout << "actual: " << static_cast<int>(arma::vectorise(val_labels[i]).index_max()) << endl;

		if (static_cast<int>(arma::vectorise(predictions[i]).index_max())
				== static_cast<int>(arma::vectorise(val_labels[i]).index_max())) {
			++total_correct;
		}
	}

	cout << "fraction correct: "
			<< static_cast<double>(total_correct / val_images.size()) << endl;

	*/

	arma::field<arma::Cube<double>> f(4);
	arma::Cube<double> a(1,1,2, arma::fill::randu);
	arma::Cube<double> b(1,1,2, arma::fill::randu);
	arma::Cube<double> c(1,1,2, arma::fill::randu);
	arma::Cube<double> d(1,1,2, arma::fill::randu);
	f[0]=a;
	f[1]=b;
	f[2]=c;
	f[3]=d;

	cout << f << endl;
	cout << "========================" << endl;

//	cout << f[0, 2] << endl;
	cout << f.rows(0,2) << endl;

	return 0;
}
