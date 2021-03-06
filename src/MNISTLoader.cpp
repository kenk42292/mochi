/*
 * MNISTLoader.cpp
 *
 *  Created on: Aug 17, 2016
 *      Author: ken
 */

#include "MNISTLoader.hpp"

using std::string;

MNIST_Loader::MNIST_Loader() {

}

MNIST_Loader::~MNIST_Loader() {
}

arma::field<arma::Cube<double> > MNIST_Loader::load_images(string images_path) {
	ifstream inStream(images_path);

	std::vector<arma::Cube<double>> result;

	if (inStream.is_open()) {
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;

		inStream.read((char*) &magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);
		inStream.read((char*) &number_of_images, sizeof(number_of_images));
		number_of_images = reverseInt(number_of_images);
		inStream.read((char*) &n_rows, sizeof(n_rows));
		n_rows = reverseInt(n_rows);
		inStream.read((char*) &n_cols, sizeof(n_cols));
		n_cols = reverseInt(n_cols);

		cout << "n_rows: " << n_rows << endl;
		cout << "n_cols: " << n_cols << endl;
		cout << "number_of_images: " << number_of_images << endl;

		for (int i = 0; i < number_of_images; ++i) {
			arma::Col<double> single_image(n_rows * n_cols, arma::fill::zeros);
			for (int r = 0; r < n_rows; ++r) {
				for (int c = 0; c < n_cols; ++c) {
					unsigned char pixel = 0;
					inStream.read((char*) &pixel, sizeof(pixel));
					single_image[r * n_rows + c] = static_cast<double>(pixel);
				}
			}
			single_image /= 255.0;
			result.push_back(arma::Cube<double>((const double*) single_image.begin(), 1, 1, single_image.size()));
		}
		inStream.close();
	} else {
		cout << "couldn't open" << endl;
	}
//	std::cout << result[0] << std::endl;
	arma::field<arma::Cube<double>> resultCubes(result.size());
	for (unsigned int i=0; i<result.size(); ++i) {
		resultCubes[i] = result[i];
	}
	return resultCubes;
}

arma::Cube<double> MNIST_Loader::charToCube(unsigned char c) {
	arma::Cube<double> vec(1, 1, 10, arma::fill::zeros);
	vec[c] = 1.0;
	return vec;
}

arma::field<arma::Cube<double>> MNIST_Loader::load_labels(string labels_path) {
	ifstream inStream(labels_path);

	std::vector<unsigned char> result_chars;

	if (inStream.is_open()) {
		int magic_number = 0;
		int number_of_labels = 0;

		inStream.read((char*) &magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);
		inStream.read((char*) &number_of_labels, sizeof(number_of_labels));
		number_of_labels = reverseInt(number_of_labels);

		cout << "number_of_labels: " << number_of_labels << endl;

		for (int i = 0; i < number_of_labels; ++i) {
			unsigned char label = 0;
			inStream.read((char*) &label, sizeof(label));
			result_chars.push_back(label);
		}
		inStream.close();
	} else {
		cout << "couldn't open" << endl;
	}

	arma::field<arma::Cube<double>> result(result_chars.size());
	for (unsigned int i=0; i<result_chars.size(); ++i) {
		result[i] = charToCube(result_chars[i]);
	}
	return result;
}

int MNIST_Loader::reverseInt(int i) {
	unsigned char c1, c2, c3, c4;
	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;

	return ((int) c1 << 24) + ((int) c2 << 16) + ((int) c3 << 8) + c4;
}
