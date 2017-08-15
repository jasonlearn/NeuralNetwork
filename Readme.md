# Neural Network

We build this Neural Network based of michael Nielson's neural network code at http://neuralnetworksanddeeplearning.com/.

We converted the provided Python code to Java and have created an Graphic User Interface.
The is a deep learning application, meaning it has machine learning capability and was built to read single digit numbers ( 0 to 9 ).  
We fed in over 60 thousand hand written digit for the application to learn from. 


### Introduction to Neural Network
What is a Neural Network? It is a set of algorithm which creates artificial neurons called *perceptrons*
The neuron model used in this program is caled the *sigmoid neuron*.
With machine learning, we will need to feed a large sample of data to the system.  Each data fed would
go through the system and be given a specific weight to identify what digit it is.
With more layers in the system, the higher accuracy it could achieve to identiy the digit.

### Parameters used:
* Test #:		Test number of specified conditions (ETA, Epoch, Layers, Neurons, Batch Size)
* Eta:			Machine learning rate
* Epoch:		Number of training sessions per result
* Layers:		Number of hidden layers within the network
* Neurons:		Number of nodes per layer, aka perceptrons
* Batch Size:	Mini batch size
* Results:		Result of learning accuracy ( out of 10,000 )
* Average:		The mean of result with identical parameters

### Test Results
We've conducted thousands of tests using different number of ETA, Epoch, Layers, Neurons, and Batch Size.
With the test results we've extracted from multiple tests.  We have noticed that when the machine uses the 
same network to build of to the next result ( Test A ).

In layer 1 and 2:
	- Larger batch size shows a larger growth rate in comparison from a smaller one.
	- Smaller batch size has a higher average score.
But as it reaches layer 3 and 4, smaller batch size would sometimes have a higher growth rate than a larger batch size.

The highest average count was 9595.70/10000 (95.96%) Test # 33 with ( ETA: 5, Epoch: 10, Layers: 2, Neurons: 40, 20, batch size 10)

The Max accuracy  at 96.45% in Test # 33.

We have discovered that with multiple layers, it is more optimal starting with higher number of neurons and descending to a smaller one.

With Test B, where all results were based of a new network.  The highest accuracy ever achieved was 
95.04% (Test #27, Eta: 3, Epoch: 10, Layers: 2, Neurons: 40, 20 Batch Size 10)

In conslusion of both tests,  a 2 layer (neuron 40, 20) neural network with 10 batch size and Epoch of 10 was the optimal number for the training dataset.

*Individual test results could be found in **neuarl_network_final_results** *

## Authors

* **Jason Chan** - *Initial work and completed*
* **Braden D'Eith** - *Initial work and completed*

