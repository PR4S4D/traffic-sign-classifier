# **Traffic Sign Recognition** 



**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./new_test_data/test1.jpeg "Traffic Sign 1"
[image5]: ./new_test_data/test2.png "Traffic Sign 2"
[image6]: ./new_test_data/test3.jpg "Traffic Sign 3"
[image7]: ./new_test_data/test4.png "Traffic Sign 4"
[image8]: ./new_test_data/test5.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/sLakshmiprasad/traffic-sign-classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 


<img src="./examples/original_images.png">

It is a bar chart showing how the data of number of images vs the classes.

![alt text][image1]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data between 0.1 to 0.9 to increase the accuracy.

Here are a few images after grey scaling and normalizing  -

<img src="./examples/preprocesed_images.png" />

The difference between the original data set and the augmented data set is the following ... 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution     	    |  Convolutional. Input = 32x32x1. Output = 28x28x6	|
| RELU					|												|
| Max pooling	      	|  Input = 28x28x6. Output = 14x14x6 			|
| Convolution     	    |  Convolutional. Input = 14x14x6 . Output = 10x10x16	|
| RELU					|												|
| Max pooling	      	|  Input = 10x10x16. Output = 5x5x16 			|
| Fully Connected 		|	Input = 400. Output = 120.					|
| RELU	    			|												|
| DROPOUT    			|   keep_prop = 0.5								|
| Fully Connected 		|	Input = 120. Output = 84					|
| RELU	    			|												|
| DROPOUT    			|	keep_prop = 0.5								|
| Fully Connected 		|	Input = 84. Output = 43					|
| RELU	    			|												|
| DROPOUT    			|	keep_prop = 0.5								|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a AdamOptimizer - Optimizer that implements the Adam algorithm.
The following hyperparameters were used - 
* ECOCHS = 100
* learning_rate = 0.001
* Batch_size = 100
* Dropout keep_prob for training = 0.5

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.965 
* test set accuracy of 0.939


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:


<img src="./new_test_data/test1.jpeg" alt="drawing" width="200"/>
<img src="./new_test_data/test2.png" alt="drawing" width="200"/>
<img src="./new_test_data/test3.jpg" alt="drawing" width="200"/>
<img src="./new_test_data/test4.png" alt="drawing" width="200"/>
<img src="./new_test_data/test5.jpg" alt="drawing" width="200"/>


The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image              	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|  Speed limit (60km/h) | Speed limit (60km/h)   						| 
| Right-of-way at the next intersection     				| Right-of-way at the next intersection			|
| Stop					| Stop											|
| Speed limit (50km/h) 	      			| Speed limit (50km/h)   						|
| Road Work 				            | Road Work     							    |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.              		| Speed limit (60km/h) |
| 1.15579814e-14		| Speed limit (80km/h) |
| 9.42286931e-18		| Ahead only |
| 1.18787582e-19		| End of no passing |
| 1.79161722e-22	    | End of no passing by vehicles over 3.5 metric tons |


For the second image ... 


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.                    | Right-of-way at the next intersection         |
| 3.60764293e-22		| Beware of ice/snow                            |
| 4.73857150e-24		| Road work                                     |
| 1.29538295e-26		| End of no passing by vehicles over 3.5 metric tons    |
| 1.30576625e-27	    | Priority road                                 |

For the third image ... 


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.                    | Stop                                          |
| 1.44890953e-08		| Yield                                         |
| 6.62971678e-10		| No entry                                      |
| 5.06716946e-10		| Speed limit (20km/h)                          |
| 3.51210894e-10	    | Keep right]                                   |

For the fourth image ... 


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99988914e-01		| Speed limit (50km/h)                          | 
| 1.04114797e-05		| Speed limit (80km/h)                          |
| 5.97663472e-07		| No vehicles                           	    |
| 2.38750051e-08		| Speed limit (30km/h)                          |
| 1.53832289e-08	    | Speed limit (100km/h)                         |

For the fifth image ... 


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 6.27373636e-01		| Road work                                     |
| 2.06997469e-01		| Bumpy road                                    |
| 2.64833849e-02		| Ahead only                                    |
| 2.50475667e-02		| Beware of ice/snow                            |
| 2.49955337e-02	    | Wild animals crossing                       |

