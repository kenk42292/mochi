/*
 * Loss.h
 *
 *  Created on: Dec 11, 2016
 *      Author: ken
 */

#ifndef LOSS_H_
#define LOSS_H_

class Loss {
public:
	Loss();
	virtual ~Loss();

	/** Single Sample Loss*/
	virtual double loss(arma::Col<double> output, arma::Col<double> y);
	virtual arma::Col<double> loss_prime(arma::Col<double> output, arma::Col<double> y);

	/** Batch Losses */
	virtual std::vector<double> loss(std::vector<arma::Col<double>> outputs, std::vector<arma::Col<double>> ys);
	virtual std::vector<arma::Col<double>> loss_prime(std::vector<arma::Col<double>> outputs, std::vector<arma::Col<double>> ys);

	/** Total Batch Loss */
	virtual double totalLoss(std::vector<arma::Col<double>> outputs, std::vector<arma::Col<double>> ys);
};

#endif /* LOSS_H_ */
