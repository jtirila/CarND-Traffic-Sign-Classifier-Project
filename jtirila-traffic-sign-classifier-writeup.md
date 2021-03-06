# **Traffic Sign Recognition** 

**This will be my writeup. Not really started yet, just getting prepared for the project and adding some
placeholders here.**

## Writeup by Juha-Matti Tirilä 

---

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
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points

**Addition begins: essential parts of rubric copied here**

The rubric points are: 

### Files submitted 
* The project submission includes all required files.

### Dataset exploration

Here is a basic summary of training, validation and testing data sets, including the numbers of images per each of the
traffic sign categories:


| Category | Explanation | Amount in training set | Amount in validation set | Amount in test set |
| -------- | ----------- | ---------------------: | -----------------------: | -----------------: |


Here is the code that was used to produce the table above:


```python

# FIXME

 ```


Figure FIXME contains an illustration of 9 randomly selected images from the training set.


## Intro FIXME

I experimented also with grayscaled images but ended up using the three color channels. However, to enhance image
contrast, I applied the FIXME transformation. The results of this transform can be seen in figure FIXME that contains
the processed versions of the same random image set shown in figure FIXME.

Another preprocessing technique I chose to use was training data augomentation. Specifically, I noticed that the relative
amounts of images from the different traffic sign classes varied greatly. Hence, I wrote a piece of code to even
out the distribution of images by cloning samples from smaller categories. These cloned images were then randomly rotated
and scaled so as to not end up with exact copies of the original images but rather slightly varied ones. Figure FIXME
illustrates this technique. The top row contains a selection of images from category FIXME (FIXME-explanation)
and subsequent rows some transformed versions of these images.



* The submission includes a basic summary of the data set.
* The submission includes an exploratory visualization on the dataset.

### Design and Test a Model Architecture

The model architecture is a sligthly modified version of the LeNet architecture. The LeNet implementation written
for previous phases of the curse seemed like a good starting point above all for the nice match of input dimension,
but also because of the good performance and relative simplicity of the architecture in comparison to some of the
arguably more advanced architectures.

As the loss function I used FIXME. I tried out a couple of different optimizers, namely AdamOptimizer and FIXME.
As performance of the model did not seem to really change with the change of optimizer, I decided to stick with FIXME
due to some of its nice theoretical characteristics: the additions of momentum and FIMXE seem to be a nice remedy
against getting stuck in local optima. In addition, the FIXME mitigates some corner cases where the optimizer may
get stuck in an unfortunate suboptimal plane with no obvious gradient direction.


* The submission describes the preprocessing techniques used and why these techniques were chosen.
* The submission provides details of the characteristics and qualities of the architecture, such as the type of model used, the number of layers, the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged.
* The submission describes how the model was trained by discussing what optimizer was used, batch size, number of epochs and values for hyperparameters.
* The project thoroughly discusses the approach taken for deriving and designing a model architecture fit for solving the problem given.


### Perfomance of the model



### Test a Model on New Images
* The submission includes five new German Traffic signs found on the web, and the images are visualized. Discussion is made as to any particular qualities of the images or traffic signs in the images that may be of interest, such as whether they would be difficult for the model to classify.
* The submission documents the performance of the model when tested on the captured images. The performance on the new images is compared to the accuracy results of the test set.
* The top five softmax probabilities of the predictions on the captured images are outputted. The submission discusses how certain or uncertain the model is of its predictions.

### Standout instructions:

#### Augment the Training Data

Augmenting the training set might help improve model performance. Common data
augmentation techniques include rotatiion, translation, zoom, flips, and/or
color perturbation. These techniques can be used individually or combined.

#### Analyze New Image Performance in More Detail

Calculating the accuracy on these five German traffic sign images found on the
web might not give a comprehensive overview of how well the model is
performing. Consider ways to do a more detailed analysis of model performance
by looking at predictions in more detail. For example, calculate the precision
and recall for each traffic sign type from the test set and then compare
performance on these five new images..

If one of the new images is a stop sign but was predicted to be a bumpy road
sign, then we might expect a low recall for stop signs. In other words, the
model has trouble predicting on stop signs. If one of the new images is a 100
km/h sign but was predicted to be a stop sign, we might expect precision to be
low for stop signs. In other words, if the model says something is a stop sign,
we're not very sure that it really is a stop sign.

Looking at performance of individual sign types can help guide how to better
augment the data set or how to fine tune the model.  

#### Create Visualizations of the Softmax Probabilities

For each of the five new images, create a graphic visualization of the soft-max
probabilities. Bar charts might work well.

**Addition ends: essential parts of rubric copied here**

###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your 
writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/jtirila/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?
* The size of test set is ?
* The shape of a traffic sign image is ?
* The number of unique classes/labels in the data set is ?

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because ...

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by ...

My final training set had X number of images. My validation set and test set had Y and Z number of images.

The sixth code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because ... To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer                 |     Description                               |
|:---------------------:|:---------------------------------------------:| 
| Input                 | 32x32x3 RGB image                             |
| Convolution 3x3       | 1x1 stride, same padding, outputs 32x32x64    |
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  outputs 16x16x64                 |
| Convolution 3x3       | etc.                                          |
| Fully connected       | etc.                                          |
| Softmax               | etc.                                          |
|                       |                                               |
|                       |                                               |
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used an ....

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image           |     Prediction                                      | 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign                     | Stop sign                             | 
| U-turn                        | U-turn                                |
| Yield                         | Yield                                 |
| 100 km/h                      | Bumpy Road                            |
| Slippery Road                 | Slippery Road                         |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| .60                   | Stop sign                                     | 
| .20                   | U-turn                                        |
| .05                   | Yield                                         |
| .04                   | Bumpy Road                                    |
| .01                   | Slippery Road                                 |


For the second image ... 
