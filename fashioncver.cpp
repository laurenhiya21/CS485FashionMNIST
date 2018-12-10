//Fashion MNIST Project

#include <iostream>
#include <fstream>
#include <armadillo>

//We are using the armadilo math library
using namespace arma;

//Forward declarations
void readCSV(const char * file); //Reads a given matrix into the neural network
void configureOutputMatricies(); //Sets up the output matrix for the labels
void calculateSizes();           //Calculates numberOfInputs and inputSize
void initWeightsAndBiases();     //Init the weight and bias matrices

//Runs the network (see comment above implemention for arguement descriptions)
double runNetwork(int t, double k, bool l, bool r);

//The matrices for the input
mat outputLabels;
mat inputMatrices;

//The matrix for parsing labels
mat outputMatricies;

//Sizes
const int hiddenSize = 16;
int outputSize;
int numberOfInputs;
int inputSize;

//Weights / Biases
mat inputToHiddenWeights;
mat hiddenBias;

mat hiddentoOutputWeights;
mat outputBias;

int main()
{
	//randomize the seed
	arma_rng::set_seed_random();

	//Read the data files into the nueral network
	readCSV("mini_train.csv");

	//Configure the label matrix
	configureOutputMatricies();

	//Calculate the sizes (input, output)
	calculateSizes();

	//initalize the weights and biases
	initWeightsAndBiases();

	//run the neural network with 200 iterations
	runNetwork(400, 2, true, false);

	//run a test to see how well it learned
	double correct = runNetwork(1, 2, false, true);

	std::cout << "The network correctly identified: " << correct << "%!\n";

	//Wait for user input so the program doesn't just clos
	getchar();

	return 0;
}

//Reads the CSV file and saves it as a matrix
void readCSV(const char * file)
{
	std::cout << "Reading: " << file << "...\n";

	//open the file
	inputMatrices.load(file, csv_ascii);

	//Get the labels from the matrix
	outputLabels = inputMatrices.col(1);

	// remove the labels from the matrix
	inputMatrices.shed_cols(0, 1);

	//we have to transpose the matricies for our NN format
	inplace_trans(inputMatrices);
	inplace_trans(outputLabels);

	std::cout << "file read!\n";
}

void configureOutputMatricies()
{
	outputMatricies
		<< 1 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << endr
		<< 0 << 1 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << endr
		<< 0 << 0 << 1 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << endr
		<< 0 << 0 << 0 << 1 << 0 << 0 << 0 << 0 << 0 << 0 << endr
		<< 0 << 0 << 0 << 0 << 1 << 0 << 0 << 0 << 0 << 0 << endr
		<< 0 << 0 << 0 << 0 << 0 << 1 << 0 << 0 << 0 << 0 << endr
		<< 0 << 0 << 0 << 0 << 0 << 0 << 1 << 0 << 0 << 0 << endr
		<< 0 << 0 << 0 << 0 << 0 << 0 << 0 << 1 << 0 << 0 << endr
		<< 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 1 << 0 << endr
		<< 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 1 << endr;
}

void calculateSizes()
{
	//We have a number of valid outputs equal to the columns of the outputMatrix
	outputSize = outputMatricies.n_cols;

	//The number of inputs is how many column the inputMatrix has (each column is an "image")
	//Note: This is after we transposed the matrix, so each image is in a row of the CSV file
	numberOfInputs = inputMatrices.n_cols;
	
	//The size of each input (image) is the number of rows
	inputSize = inputMatrices.n_rows;
}

void initWeightsAndBiases()
{
	//randomly init the hidden layer weights + biases
	inputToHiddenWeights = (2 * randu<mat>(hiddenSize, inputSize)) - 1;
	hiddenBias = (2 * randu<mat>(hiddenSize, 1)) - 1;

	//randomly init the output layer weights + biases
	hiddentoOutputWeights = (2 * randu<mat>(outputSize, hiddenSize)) - 1;
	outputBias = (2 * randu<mat>(outputSize, 1)) - 1;
}

//the difference between the expected output and our actual output squared
mat calcCost(const mat& t, const mat& a)
{
	//% is elementwise multiplacation
	mat retval = (t - a) % (t - a);

	return retval;
}

//the activations are just the weight * input + bias
mat netOutput(const mat& w, const mat& p, const mat& b)
{
	mat retval = (w * p) + b;

	return retval;
}

mat logsig(const mat& m)
{
	//logSig(n) = 1 / (1 + exp(-n))
	mat retval = 1 / (1 +  exp((-1 * m)) );

	return retval;
}

mat deltaLogSig(const mat& m)
{
	//d = a.*(1-a);
	mat retval = m % (1 - m);

	return retval;
}

//The network function
//----------------------------------
// t = the number of iterations to run the network
// k = weightconstant for adjusting the weights and biases
// l = if learning is enabled
// e = if we want to return what % of the inputs network correctly identified
//     Note: If r is false, the return will be the avg cost
double runNetwork(int t, double k, bool l, bool r)
{
	//init the times trained and return value to 0
	int timesTrained = 0;
	int retval = 0;

	//number of correct guesses this batch
	int numCorrect = 0;

	//we will train the network t number of times
	while (timesTrained < t)
	{
		//increment the number of times we have trained
		++timesTrained;

		//%initialize the cost of this batch
		//batchCost = zeros(outputSize, 1);

		//for each of the inputs in the batch
		for (int i = 0; i < numberOfInputs; ++i)
		{
			//get the corresponding input
			mat inputVec = inputMatrices.col(i);

			int label = outputLabels(0,i);

			//get the desired output matrix from the label
			mat desiredOutput = outputMatricies.col(label);

			//calulate the output of the hidden layer
			mat hiddenActivations = netOutput(inputToHiddenWeights, inputVec, hiddenBias);
			mat hiddenOutput = logsig(hiddenActivations);

			//use the output of the hidden layer as the inputs for the output layer
		    mat outputActivations = netOutput(hiddentoOutputWeights, hiddenOutput, outputBias);
			mat finalOutput = logsig(outputActivations);

			//calculate the error of this result
			mat error = desiredOutput - finalOutput;

			//calculate cost of this input
			mat cost = calcCost(desiredOutput, finalOutput);

			//add the error from this batch to the current total error matrix
			//batchCost = batchCost + cost;

			//if we are returning the %correct instead of the cost
			if (r == true)
			{
				double bestLabelValue = -999999;
				int bestLabel = 0;

				//figure out which label the network thought was correct
				for (int j = 0; j < outputSize; ++j)
				{
					double currentLabel = finalOutput(j,0);

					if (bestLabelValue < currentLabel)
					{
						bestLabelValue = currentLabel;
						bestLabel = j;
					}

				}

				//if the network guessed correctly, give it a point!
				if (bestLabel == label)
					numCorrect++;
			}

			//if we are not learning, no need to update the weights / bias
			if (l == false)
				continue;

			//get the delta for backprop
			mat hiddenToOutputDelta = deltaLogSig(finalOutput) % error;
			mat inputToHiddenDelta = deltaLogSig(hiddenOutput) % (hiddentoOutputWeights.t() * hiddenToOutputDelta);


			//adjust the weights of the network
		    hiddentoOutputWeights += k * hiddenToOutputDelta * hiddenOutput.t();
			inputToHiddenWeights  += k * inputToHiddenDelta  * inputVec.t();

			//adjust the bais of the nextwork
			outputBias = outputBias + k * hiddenToOutputDelta;
			hiddenBias = hiddenBias + k * inputToHiddenDelta;

		}

	}

	return (double)numCorrect / (double)( t * numberOfInputs) * 100;
}