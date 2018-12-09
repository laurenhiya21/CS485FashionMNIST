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

//The matrices for the input
mat outputLabels;
mat inputMatrices;

//The matrix for parsing labels
mat outputMatricies;

//Sizes
const int hiddenSize = 8;
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
	//Read the data files into the nueral network
	readCSV("mini_train.csv");

	//Configure the label matrix
	configureOutputMatricies();

	//Calculate the sizes (input, output)
	calculateSizes();

	//initalize the weights and biases
	initWeightsAndBiases();

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

//The network function
//----------------------------------
// t = the number of iterations to run the network
// k = weightconstant for adjusting the weights and biases
// l = if learning is enabled
// e = if we want to return what % of the inputs network correctly identified
//     Note: If r is false, the return will be the avg cost
double runNetwork(int t, double k, bool l, bool r)
{

	return 0;
}