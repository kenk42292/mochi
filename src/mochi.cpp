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
using namespace std;



int main() {
	cout << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!

	arma::Mat<double> a(2, 2, arma::fill::randu);
	arma::Mat<double> b(2, 2, arma::fill::randu);



	std::vector<arma::Mat<double>> zs(2);
//	zs.push_back(a);
//	zs.push_back(b);
	zs[0] = a;
	zs[1] = b;


	cout << zs[1] << endl;

	Sigmoid s;
	std::vector<arma::Mat<double>> ys = s.feedForward(zs);

	cout << ys[1] << endl;

	return 0;
}
