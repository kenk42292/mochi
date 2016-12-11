/*
 * NeuralNet.h
 *
 *  Created on: Dec 8, 2016
 *      Author: ken
 */

#ifndef NEURALNET_H_
#define NEURALNET_H_

#include "loss/Loss.h"
#include "layer/Layer.h"

class NeuralNet {
private:
	std::vector<Layer> layers;
	Loss loss;
	unsigned int batchSize;
public:
	NeuralNet();
	virtual ~NeuralNet();

	/** */
	void train(const std::vector<arma::Col<double>>& inputs, const std::vector<arma::Col<double>>& outputs);
	void train(const std::vector<arma::Mat<double>>& inputs, const std::vector<arma::Col<double>>& outputs);
	void train(const std::vector<arma::Cube<double>>& inputs, const std::vector<arma::Col<double>>& outputs);

};

#endif /* NEURALNET_H_ */
