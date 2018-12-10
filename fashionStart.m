%the inputs of the neural network
%----------------------------------

%the matricies for the inputs + labels
global csvLabels;
global csvInput;
global outputMatricies;
global outputSize;
%the number of inputs
global numberOfInputs;
global trainData;
global latestPrediction;

latestPrediction = 0;

beginTime = tic;

disp("Reading: train.csv..." );

%get the inputs and labels from MNIST
trainData = csvread('train.csv',1,1);
   
timeElapsed = toc(beginTime);

disp("file read! Took " + timeElapsed + "seconds!");

csvLabels = trainData(:,1);
   
trainData(:,1) = [];
    
csvInput = trainData;
    
%transpose the input
csvInput = csvInput';
    
%normalize the input
csvInput = csvInput./255;
    
%the matrix represenation of each label
t0 = [1;0;0;0;0;0;0;0;0;0];
t1 = [0;1;0;0;0;0;0;0;0;0];
t2 = [0;0;1;0;0;0;0;0;0;0];
t3 = [0;0;0;1;0;0;0;0;0;0];
t4 = [0;0;0;0;1;0;0;0;0;0];
t5 = [0;0;0;0;0;1;0;0;0;0];
t6 = [0;0;0;0;0;0;1;0;0;0];
t7 = [0;0;0;0;0;0;0;1;0;0];
t8 = [0;0;0;0;0;0;0;0;1;0];
t9 = [0;0;0;0;0;0;0;0;0;1];

%put them all in a single matrix
outputMatricies = [t0 t1 t2 t3 t4 t5 t6 t7 t8 t9];
    
%Transposing outputMatricies
outputMatricies = outputMatricies';
    
%Uncomment this line to verify that csvLabels is working correctly
%disp(csvLabels);
 
%Uncomment this line to verify that csvInput is working correctly
%disp(csvInput) 

%parameters neural network
%----------------------------------

%The size (neurons) of the hidden layer
global hiddenSize;
hiddenSize = 8;

global batchSize;
batchSize = 500;

global batching;
batching = true;

global maxInputs;
maxInputs = 10000;

%the size of the neural network (autogenerate later, hardcode for now)
%------------------------
%The size of the output layer (0,1 and 2 would be 3 possible outcomes)

outputSize = size(outputMatricies,2);

numberOfInputs = size(csvInput,2);

if numberOfInputs > maxInputs
    numberOfInputs = maxInputs;
end

%The size of the input vector
global inputSize;
inputSize = size(csvInput,1);

%intialize the neural network
%----------------------------------

%create the weight matrix for the input -> hidden layer
%initialize the values to be between -1 and 1 
global inputToHiddenWeights;
inputToHiddenWeights = (2).*rand(hiddenSize,inputSize) - 1;

%create the bias vector for the hidden layer
global hiddenBias;
hiddenBias = (2).*rand(hiddenSize,1) - 1;

%create the weight matrix for the hidden -> output layer
%initialize the values to be between -1 and 1 
global hiddentoOutputWeights;
hiddentoOutputWeights = (2).*rand(outputSize,hiddenSize) - 1;

%create the bias vector for the output layer
%initialize the values to be between -1 and 1 
global outputBias;
outputBias = (2).*rand(outputSize,1) - 1;

%for the visuals
%----------------------------------
figure; 
hold on; 
grid on;
ylabel("Cost");
xlabel("Iteration");

%run the network
%----------------------------------

%A variable that represents total runtime of code
BeginTime = tic;

%run the neural network with 200 iterations
runNetwork(200, 2, true, true, false);
    
%run a test to see how well it learned
correct = runNetwork(1, 2, false, false, true);
    
title(hiddenSize + " Neuron with MNIST Data " + correct + "% Correct");

%show the output
disp("The network correctly identified: " + correct + "%!");

%Get total runtime in seconds
EndTime = toc(BeginTime);

%convert to milliseconds
EndTime = EndTime * 1000;

disp("Run Time: " + EndTime + "milliseconds!");

%the network function
%----------------------------------
% t = the number of iterations to run the network
% k = weightconstant for adjusting the weights and biases
% l = if learning is enabled
% g = if we should graph the results
% r = if we want to report what % of the inputs network correctly identified 
%     (returned as c), Note: If r is false, c will be the avg cost
function c = runNetwork(t, k, l, g, r)

    %get the input matricies
    global csvInput;
    global csvLabels;
    global outputMatricies;
    
    %get the weights
    global inputToHiddenWeights;
    global hiddentoOutputWeights;
    
    %get the biases
    global hiddenBias;
    global outputBias;
    
    %We need to get the sizes
    global numberOfInputs;
    global outputSize;
    global hiddenSize;
    global inputSize;
    
    %For batching
    global batching;
    global batchSize;
    
    %init the times trained and cost to 0
    timesTrained = 0;
    c = 0;
    
    %Average run time variable
    averageRunTime = 0;
    
    iterationTime = tic;
    
    currentBatchSize = numberOfInputs;
    
    if batching == true
        currentBatchSize = batchSize;
    end
  
    
    while timesTrained < t
        %This one is similar to begintime, but it will be used exclusively to
        %measure time for every iteration
         
        timesTrained = timesTrained + 1;
        
        %initialize the cost of this batch
        batchCost = zeros(outputSize,1);
        
        %batch deltas for... batching
        batchOutputDeltas = zeros(outputSize,hiddenSize);
        batchHiddenDeltas = zeros(hiddenSize,inputSize);
        
        Error = zeros(outputSize,1);
        
        %number of correct guesses this batch
        numCorrect = 0;
        currentBatchCalculated = 0;
 
        %for each of the inputs in the set
        for i = 1:(numberOfInputs)
            %increment cbc
            currentBatchCalculated = currentBatchCalculated + 1;
            
            %get the corresponding input
            inputVec = csvInput(:,i);
          
            %get the label of the output
            label = csvLabels(i,:) + 1; 
   
            %get the desired output matrix from the label
            desiredOutput = outputMatricies(:,label);
          
            %calulate the output of the hidden layer
            hiddenActivations = netOutput(inputToHiddenWeights,inputVec,hiddenBias);
            hiddenOutput = logsig(hiddenActivations);
       
            %use the output of the hidden layer as the inputs for the output layer
            outputActivations = netOutput(hiddentoOutputWeights,hiddenOutput,outputBias);
            finalOutput = logsig(outputActivations);
       
            latestPrediction = finalOutput;
            
            %calculate the error of this result
            error = desiredOutput - finalOutput;
    
            %calculate the error (cost) of this input
            cost = calcCost(desiredOutput, finalOutput);
          
            %add the error from this batch to the current total error matrix
            batchCost = batchCost + cost;
            
            %if we are returning the %correct instead of the cost
            if r == true
                
                bestLabelValue = -999999;
                bestLabel = 1;
                %figure out which label the network thought was correct
                for j = 1:outputSize
                    currentLabel = finalOutput(j);
                    
                    if bestLabelValue < currentLabel
                        bestLabel = j;
                        bestLabelValue = currentLabel;
                    end
                end
                
                %if the network guessed correctly, give it a point!
                if bestLabel == label
                    numCorrect = numCorrect + 1;
                end

            end
            
            %if we are not learning, no need to update the weights / bias
            if l == false
                continue;
            end
            
            %get the delta for backprop
            hiddenToOutputDelta = deltaLogSig(finalOutput).*error;
            inputToHiddenDelta = deltaLogSig(hiddenOutput).*(hiddentoOutputWeights.'*hiddenToOutputDelta);
            
            finalHiddenToOutputDelta = k.*hiddenToOutputDelta*(hiddenOutput.');
            finalInputToHiddenWeights = k.*inputToHiddenDelta*(inputVec.');
            
            if batching == true
                batchOutputDeltas = batchOutputDeltas + finalHiddenToOutputDelta;
                batchHiddenDeltas = batchHiddenDeltas + finalInputToHiddenWeights;
                
                if currentBatchCalculated ~= batchSize && i ~= numberOfInputs
                    continue;
                end
               
                %avarage the deltas
                batchOutputDeltas = batchOutputDeltas./currentBatchCalculated;
                batchHiddenDeltas = batchHiddenDeltas./currentBatchCalculated;
               
                %these are the deltas we will use
                finalHiddenToOutputDelta = batchOutputDeltas;
                finalInputToHiddenWeights = batchHiddenDeltas;
                
                %reset the batch deltas
                batchOutputDeltas = zeros(outputSize,hiddenSize);
                batchHiddenDeltas = zeros(hiddenSize,inputSize);

                %reset cbc
                currentBatchCalculated = 0;
            end
            
            
            
            %adjust the weights of the network
            hiddentoOutputWeights = hiddentoOutputWeights + finalHiddenToOutputDelta;
            inputToHiddenWeights = inputToHiddenWeights + finalInputToHiddenWeights;
            
            %adjust the bais of the nextwork
            outputBias = outputBias + k.*hiddenToOutputDelta;
            hiddenBias = hiddenBias + k.*inputToHiddenDelta;
        
        %Get Time of current iteration using iteration time

        end
        
        %avarage the cost
        avgCost = batchCost / numberOfInputs;
       
        %the cost is the sum of this batches costs
        cost = sum(avgCost);
        
        if g == true
        %plot the cost over the itterations
        plot(timesTrained,cost,'*');
        end
        
        %update the cost
        c = c + cost;
 
    end

    c = c / t;
    
    if r == true
        c = (numCorrect / (t * numberOfInputs) * 100);
    end 

    currentIteration = toc(iterationTime);
        
    %Convert from seconds to milliseconds
    currentIteration = currentIteration * 1000;
        
    %Add it to average time variable
    averageRunTime = averageRunTime + currentIteration;

    %Average the total time by dividing by number of times loop runs
    averageRunTime = averageRunTime / t;

    %disp("Average Run Time: "); 
    %fprintf('%d milliseconds\n',averageRunTime);
    

    %disp("Prediction: ");
    %disp(latestPrediction);
end

%delta of the log sig function
function d = deltaLogSig(a)
d = a.*(1-a);
end

%the difference between the expected output and our actual output squared
function c = calcCost(t,a)
c = (t-a).*(t-a);
end

% the activations are just the weight * input + bias
function n = netOutput(w,p,b)
n = (w * p) + b;
end
