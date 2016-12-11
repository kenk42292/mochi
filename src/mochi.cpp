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

	arma::Cube<double> a(1, 4, 1, arma::fill::randu);
	arma::Mat<double> m(2, 4, arma::fill::randu);
	arma::Col<double> v = arma::vectorise(a);

	cout << m*v << endl;

	arma::Mat<double> p = m*v;

	arma::cube res((const double*) p.begin(), 1, 1, 2);
	cout << res << endl;


	return 0;
}
