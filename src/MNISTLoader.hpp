/*
 * MNISTLoader.h
 *
 *  Created on: Aug 17, 2016
 *      Author: ken
 */

#ifndef MNISTLOADER_HPP_
#define MNISTLOADER_HPP_

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <armadillo>
using std::ifstream;
using std::string;
using std::cout;
using std::endl;

class MNIST_Loader {
private:
	int reverseInt(int i);
public:
	MNIST_Loader();
	virtual ~MNIST_Loader();
	arma::Cube<double> charToCube(unsigned char c);
	arma::field<arma::Cube<double>> load_images(string images_path);	// Return 784 long cubes
	arma::field<arma::Cube<double>> load_labels(string labels_path);	// Return label as 10 long cubes
};

#endif /* MNISTLOADER_HPP_ */
