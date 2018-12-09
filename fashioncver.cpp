//Fashion MNIST Project

#include <iostream>
#include <fstream>
#include <armadillo>

//We are using the armadilo math library
using namespace arma;

//Forward declarations
void readCSV(const char * file); //Reads a given matrix into the neural network
void configureOutputMatricies(); //Sets up the output matrix for the labels

//The matricies for the input
mat outputLabels;
mat inputMatrices;

//The matrix for parsing labels
mat outputMatricies;

//Sizes
const int hiddenSize = 8;
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

	configureOutputMatricies();

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