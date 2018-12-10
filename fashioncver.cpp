//Fashion MNIST Project

#include <iostream>
#include <fstream>
#include <armadillo>
#include <Windows.h>

//We are using the armadilo math library
using namespace arma;

//Forward declarations
void readCSV(const char * file); //Reads a given matrix into the neural network
void readTest(const char * file); //reads the test file
void configureOutputMatricies(); //Sets up the output matrix for the labels
void calculateSizes();           //Calculates numberOfInputs and inputSize
void initWeightsAndBiases();     //Init the weight and bias matrices

//Runs the network (see comment above implemention for arguement descriptions)
double runNetwork(int t, double k, bool l, bool r, bool test);

//The matrices for the input
mat outputLabels;
mat inputMatrices;
mat outputSubmission;

//The matrix for parsing labels
mat outputMatricies;

//Parameters
const int kValue = 1.5;
const int hiddenSize = 8;

const int hiddenLayers = 1;
const int hidden2Size = 1;

int iterations = 300;

bool batching = true;
int batchSize = 30;

//SHOULD BE 0 and TRUE for final test
int maxInputs = 0;
bool output = true;

//Sizes
int outputSize;
int numberOfInputs;
int inputSize;

//Weights / Biases
mat inputToHiddenWeights;
mat hiddenBias;

mat hiddentoHidden2Weights;
mat hidden2Bias;

mat hiddentoOutputWeights;
mat outputBias;

//For tracking time
DWORD totalTime;

int main()
{
	//randomize the seed
	arma_rng::set_seed_random();

	//Read the data files into the nueral network
	readCSV("train.csv");

	//Configure the label matrix
	configureOutputMatricies();

	//Calculate the sizes (input, output)
	calculateSizes();

	//initalize the weights and biases
	initWeightsAndBiases();

	//run the neural network with 200 iterations
	runNetwork(iterations, kValue, true, false, false);

	double trainedTime = totalTime;

	//run a test to see how well it learned
	double correct = runNetwork(1, 2, false, true, false);

	if (hiddenLayers == 2)
		std::cout << hiddenSize << "\\" << hidden2Size << " Neurons, 2 Hidden Layers, " << correct << "% Correct\n";
	else
		std::cout << hiddenSize << " Neurons, 1 Hidden Layer, " << correct << "% Correct\n";

	if (batching == false)
	{
		std::cout << "No batching, ";
	}	
	else
	{
		std::cout << "Batches of " << batchSize << ", ";
	}

	std::cout << "Run Time: " << (double)trainedTime / 1000.0 << " seconds\n";

	std::cout << iterations << " Iterations\n";

	if (output == false)
	{
		getchar();
		return 0;
	}

	readTest("test.csv");

	calculateSizes();

	//test time
	runNetwork(1, 2, false, true, true);

	outputSubmission.save("ourSubmission.csv", csv_ascii);

	std::cout << "output: ourSubmission.csv\n";

	//Wait for user input so the program doesn't just close
	getchar();

	return 0;
}

//Reads the CSV file and saves it as a matrix
void readCSV(const char * file)
{
	std::cout << "Reading: " << file << "...\n";

	DWORD startTime = timeGetTime();

	//open the file
	inputMatrices.load(file, csv_ascii);

	//Get the labels from the matrix
	outputLabels = inputMatrices.col(1);

	// remove the labels from the matrix
	inputMatrices.shed_cols(0, 1);

	//we have to transpose the matricies for our NN format
	inplace_trans(inputMatrices);
	inplace_trans(outputLabels);

	DWORD endTime = timeGetTime();

	std::cout << "file read! Took " << (double)(endTime - startTime) / 1000.0 << " seconds!\n";
}

//Reads the CSV file and saves it as a matrix
void readTest(const char * file)
{
	std::cout << "Reading Test: " << file << "...\n";

	DWORD startTime = timeGetTime();

	//open the file
	inputMatrices.load(file, csv_ascii);

	//Get the labels from the matrix
	outputLabels = inputMatrices.col(0);
	outputSubmission = outputLabels;

	//setup labels / submission
	outputLabels.zeros();
	outputSubmission.resize(outputSubmission.n_rows, 2);

	// remove the labels from the matrix
	inputMatrices.shed_col(0);

	//we have to transpose the matricies for our NN format
	inplace_trans(inputMatrices);
	inplace_trans(outputLabels);

	DWORD endTime = timeGetTime();

	std::cout << "file read! Took " << (double)(endTime - startTime) / 1000.0 << " seconds!\n";
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

	//cap the number of imputs for testing and performance
	if (maxInputs > 0 && numberOfInputs > maxInputs)
	{
		numberOfInputs = maxInputs;
	}
	
	//The size of each input (image) is the number of rows
	inputSize = inputMatrices.n_rows;
}

void initWeightsAndBiases()
{
	//randomly init the hidden layer weights + biases
	inputToHiddenWeights = (2 * randu<mat>(hiddenSize, inputSize)) - 1;
	hiddenBias = (2 * randu<mat>(hiddenSize, 1)) - 1;

	if (hiddenLayers == 2)
	{
		//randomly init the 2nd hidden layer weights + biases
		hiddentoHidden2Weights = (2 * randu<mat>(hidden2Size, hiddenSize)) - 1;
		hidden2Bias = (2 * randu<mat>(hidden2Size, 1)) - 1;

		//randomly init the output layer weights + biases
		hiddentoOutputWeights = (2 * randu<mat>(outputSize, hidden2Size)) - 1;
		outputBias = (2 * randu<mat>(outputSize, 1)) - 1;
	}
	else if (hiddenLayers == 1)
	{
		//randomly init the output layer weights + biases
		hiddentoOutputWeights = (2 * randu<mat>(outputSize, hiddenSize)) - 1;
		outputBias = (2 * randu<mat>(outputSize, 1)) - 1;
	}

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
	mat retval = 1.0 / (1.0 +  exp((-1.0 * m)) );

	return retval;
}

mat deltaLogSig(const mat& m)
{
	//d = a.*(1-a);
	mat retval = m % (1.0 - m);

	return retval;
}

//The network function
//----------------------------------
// t = the number of iterations to run the network
// k = weightconstant for adjusting the weights and biases
// l = if learning is enabled
// e = if we want to return what % of the inputs network correctly identified
//     Note: If r is false, the return will be the avg cost
// test = if this should save results to a test file
double runNetwork(int t, double k, bool l, bool r, bool test)
{
	//init the times trained and return value to 0
	int timesTrained = 0;
	int retval = 0;

	//number of correct guesses this batch
	int numCorrect = 0;

    totalTime = 0;

	int currentBatchSize = numberOfInputs;

	if (batching == true)
	{
		currentBatchSize = batchSize;
	}

	DWORD startTime = timeGetTime();

	mat batchOutputDeltas;
	mat batchHidden2Deltas;
	mat batchHiddenDeltas;

	mat batchOutputBias(outputSize,1);
	mat batchHidden2Bias(hidden2Size,1);
	mat batchHiddenBias(hiddenSize,1);

	double nextUpdate = 0;

	//we will train the network t number of times
	while (timesTrained < t)
	{
		//increment the number of times we have trained
		++timesTrained;

		//init the current batch calculated
		int currentBatchCalculated = 0;

		if (hiddenLayers == 2)
		{
			batchOutputDeltas.zeros(outputSize, hidden2Size);
			batchHidden2Deltas.zeros(hidden2Size, hiddenSize);
			batchHiddenDeltas.zeros(hiddenSize, inputSize);
		}
		else if (hiddenLayers == 1)
		{
			batchOutputDeltas.zeros(outputSize, hiddenSize);
			batchHiddenDeltas.zeros(hiddenSize, inputSize);
		}

		batchOutputBias.zeros();
		batchHidden2Bias.zeros();
	    batchHiddenBias.zeros();

		//for each of the inputs in the batch
		for (int i = 0; i < numberOfInputs; ++i)
		{
			//We have calculated another input from this batch
			currentBatchCalculated++;

			//get the corresponding input
			mat inputVec = inputMatrices.col(i);

			// divide the input vector by 255 to normalize the inputs
			inputVec /= 255;

			int label = outputLabels(0, i);

			//get the desired output matrix from the label
			mat desiredOutput = outputMatricies.col(label);

			//calulate the output of the hidden layer
			mat hiddenActivations = netOutput(inputToHiddenWeights, inputVec, hiddenBias);
			mat hiddenOutput = logsig(hiddenActivations);

			//use the output of the hidden layer as the inputs for the output layer
			mat outputActivations;
			mat finalOutput;

			//If we have 2 layers
			mat hidden2Activations;
			mat hidden2Output;

			if (hiddenLayers == 2)
			{
				hidden2Activations = netOutput(hiddentoHidden2Weights, hiddenOutput, hidden2Bias);
				hidden2Output = logsig(hidden2Activations);

				outputActivations = netOutput(hiddentoOutputWeights, hidden2Output, outputBias);
				finalOutput = logsig(outputActivations);
			}
			else if (hiddenLayers == 1)
			{
				//use the output of the hidden layer as the inputs for the output layer
				outputActivations = netOutput(hiddentoOutputWeights, hiddenOutput, outputBias);
				finalOutput = logsig(outputActivations);
			}

			//calculate the error of this result
			mat error = desiredOutput - finalOutput;

			//calculate cost of this input
			mat cost = calcCost(desiredOutput, finalOutput);

			//if we are returning the %correct instead of the cost
			if (r == true)
			{
				double bestLabelValue = -999999;
				int bestLabel = 0;

				//figure out which label the network thought was correct
				for (int j = 0; j < outputSize; ++j)
				{
					double currentLabel = finalOutput(j, 0);

					if (bestLabelValue < currentLabel)
					{
						bestLabelValue = currentLabel;
						bestLabel = j;
					}

				}

				//if the network guessed correctly, give it a point!
				if (bestLabel == label)
					numCorrect++;

				if (test == true)
				{
					outputSubmission(i, 1) = bestLabel;
				}

			}

			//if we are not learning, no need to update the weights / bias
			if (l == false)
				continue;

			//forward dec the matrices
			mat hiddenToOutputDelta;
			mat hiddenToHidden2Delta;
			mat inputToHiddenDelta;
			mat finalHiddenToOutputDelta;
			mat finalhiddenToHidden2Delta;
			mat finalInputToHiddenDelta;

			if (hiddenLayers == 1)
			{
				//get the delta for backprop
				hiddenToOutputDelta = deltaLogSig(finalOutput) % error;
				inputToHiddenDelta = deltaLogSig(hiddenOutput) % (hiddentoOutputWeights.t() * hiddenToOutputDelta);

				//adjust the weights of the network
				finalHiddenToOutputDelta = k * hiddenToOutputDelta * hiddenOutput.t();
				finalInputToHiddenDelta = k * inputToHiddenDelta  * inputVec.t();
			}
			else
			{
				//get the delta for backprop
				hiddenToOutputDelta = deltaLogSig(finalOutput) % error;
				hiddenToHidden2Delta = deltaLogSig(hidden2Output) % (hiddentoOutputWeights.t() * hiddenToOutputDelta);
				inputToHiddenDelta = deltaLogSig(hiddenOutput) % (hiddentoHidden2Weights.t() * hiddenToHidden2Delta);

				//adjust the weights of the network
				finalHiddenToOutputDelta = k * hiddenToOutputDelta * hidden2Output.t();
				finalhiddenToHidden2Delta = k * hiddenToHidden2Delta * hiddenOutput.t();
				finalInputToHiddenDelta = k * inputToHiddenDelta  * inputVec.t();
			}


			if (batching == true)
			{
				batchOutputDeltas += finalHiddenToOutputDelta;
				batchHiddenDeltas += finalInputToHiddenDelta;

				if (hiddenLayers == 2)
				{
					batchHidden2Deltas += finalhiddenToHidden2Delta;
					batchHidden2Bias += hiddenToHidden2Delta;
				}
					
				batchOutputBias += hiddenToOutputDelta;
				batchHiddenBias += inputToHiddenDelta;

				if (currentBatchCalculated != batchSize && i != numberOfInputs)
					continue;

				//avarage the deltas
				batchOutputDeltas /= currentBatchCalculated;
				batchHiddenDeltas /= currentBatchCalculated;
				batchHidden2Deltas /= currentBatchCalculated;

				batchOutputBias /= currentBatchCalculated;
				batchHidden2Bias /= currentBatchCalculated;
				batchHiddenBias /= currentBatchCalculated;

				//these are the deltas we will use
				finalHiddenToOutputDelta = batchOutputDeltas;
				finalInputToHiddenDelta = batchHiddenDeltas;
				finalhiddenToHidden2Delta = batchHidden2Deltas;

				hiddenToOutputDelta = batchOutputBias;
				hiddenToHidden2Delta = batchHidden2Bias;
				inputToHiddenDelta = batchHiddenBias;

				//reset the batch deltas
				batchOutputDeltas.zeros();
				batchHiddenDeltas.zeros();
				batchHidden2Deltas.zeros();

				batchOutputBias.zeros();
				batchHidden2Bias.zeros();
				batchHiddenBias.zeros();

				//reset cbc
				currentBatchCalculated = 0;
			}

			//adjust the weights of the network
		    hiddentoOutputWeights += finalHiddenToOutputDelta;
			inputToHiddenWeights  += finalInputToHiddenDelta;

			//adjust the bais of the nextwork
			outputBias = outputBias + k * hiddenToOutputDelta;
			hiddenBias = hiddenBias + k * inputToHiddenDelta;

			if (hiddenLayers == 2)
			{
				hiddentoHidden2Weights += finalhiddenToHidden2Delta;
				hidden2Bias = hidden2Bias + k * hiddenToHidden2Delta;
			}

		}

		if (l == false)
			continue;



		double percentTrained = (double)(timesTrained + 1) / (double)(t) * 100;

		if (percentTrained > nextUpdate)
		{
			int intUpdate = (int)nextUpdate;
			
			if (intUpdate % 10 == 0)
			{
				cout << intUpdate;
			}
			else
			{
				cout << ".";
			}

			if (intUpdate == 100)
			{
				cout << endl;
			}

			nextUpdate += 2.5;
		}
	}

	//Calculate the total time
	DWORD endTime = timeGetTime();
	totalTime = endTime - startTime;

	return (double)numCorrect / (double)( t * numberOfInputs) * 100;
}