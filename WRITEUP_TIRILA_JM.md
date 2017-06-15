# Traffic Sign Recognition

##Writeup, J-M Tirilä 

---

### Overall goals of the project

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

### Rubric Points


I will now consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

#### A Basic Summary of the Data Set 

##### Test train split

The traffic sign image set that I downloaded was already divided into a training, validation and testing set. I did not 
change this even though it could be beneficial to combine the sets and perform the splits from scratch, using e.g. 
the `train_test_split` function of `scikit-learn`. 

##### Statistics and visualizations 

The traffic sign data I loaded was already divided into training, validation and test sets and I did not change this. 
So the 

The code for this step is contained in the second code cell of the IPython notebook. First I print out the 
following statistics about the data set, using just plain Python to to calculate some summary statistics of the traffic
signs data set.

* The size of training set is ?
* The size of test set is ?
* The shape of a traffic sign image is ?
* The number of unique classes/labels in the data set is ?

#### Some visualizations of the data set 

The code for this step is contained in the FIXME:third code cell of the IPython notebook. The bulk of the work is 
described in the cell and I won't repeat it here. To summarize, I plotted a bar chart containing the numbers of 
examples in each class, and then some examples of each class. 

As a first preprocessing step, I augment the data so that each class is approximately equally represented in the
training data set. This could be done by just copying some images over and over, but I figured it is better to introduce
"new" examples by performing some slight alterations of the images while copying. 

Hence, I applied some random rotations and scalings to the images. FIXME: Below is an example of a traffic sign 
image and its altered version. 

With this augmentation technique, I ended up with a data set of FIXME examples. 

#### Luminosity normalization

Looking at the visualizations, it is obvious that the images vary greatly in luminosity and contrast. I hence looked 
for a method to enhance the contrast and also unify the image luminosity / brighness across examples. The method 
I ended up using is to perform histrogram normalization on the Y channel of the image, temporarily first converted 
into the YUV colorspace format for this procedure. The code to do this using OpenCV can be found below. 

```python
import cv2
import numpy as np

img = cv2.imread('input.jpg')

img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

# equalize the histogram of the Y channel
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

# convert the YUV image back to RGB format
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

```

#### Shuffling Before Training and Between Epochs

To eliminate all possible effects related to training data ordering, I make sure to shuffle the training examples and 
the corresponding labels every time before a new batch of training is about to happen. This is peformed at multiple 
locations in the code. The shuffling is performed using the `shuffle` function in `sklean.utils`, and it is used as 
follows. 

```python
from sklearn.utils import shuffle
features_train, labels_train = shuffle(features_train, labels_train)

```



https://www.packtpub.com/mapt/book/application_development/9781785283932/2/ch02lvl1sec26/enhancing-the-contrast-in-an-image
https://stackoverflow.com/a/38312281

The actual code I use is a bit different due to different order of colorspace conversions. The code can be found in 
cell FIXME of the notebook. 


![alt text][image1]

### The Model Architecture Design

The model architecture I ended up using is just the same as in the instructions. I played around 
with some simpler feedforward network architectures also, and tried adding layers to the LeNet architecture, 
but the improvements were not worth the increased computing demands. 

Here is an image of the LeNet architecture: 

![Lenet architecture][lenet_arch_image]

The one alteration I did make to the LeNet model, however, was the addition of dropout as a regularization technique. 

I tried adding dropout to various layers in the network, but it turned out it worked best just after the second 
pooling layer. I am not sure why the effect would be different here compared to performing dropout after flattening 
the output of pooling, but this was my experience anyhow. 

The code that defines the LeNet architecture can be found in cell FIXME of the notebook, in the LeNet function. 
For reasons explained later, I also update a network parameter dict when defining the architecture. 
That is why in addition to the output layer, also the network dict is returned from the function. 

The LeNet architecture consists of the folowing layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 
 
#### The hyperparameters

I concluded that the number of epochs should not be too large, so I ended up using 10. 

For the learning rate, I figured something rather small is appropriate. Testing various values, 
I settled on `0.0014`. 

A further thing to consider are the distributions of the initial weights. I am using random normal weights. 
My experience is that the choice of variance value used to draw these initial weigts is actually pretty important. 

After playing around with different sigmas, I ended up using FIXME: sigma. 

### Training and evaluating the model

The training code I used was basically copied over from the instruction videos and the labs preceding this project. 
The training function can be seen in code cell #FIXME

After each epoch, some metrics are printed out concerning the performance of the model. Here is an example output: 

```
2017-06-15 10:57:41.799758: EPOCH 1 ...
Validation Accuracy = 0.887302
```

The evaluation code is written in the code cell FIXME of the notebook. 


### Iterating to come up With a Functional Pipeline and Good Hyperparameters

Of course, the LeNet code or the choices of hyperparameters listed above needed to be revised multiple times, 
and the ones listed above are just the final choices I made. The process involved trying out different 
combinations of hyperparameter values and fixing minor bugs in the processing pipeline. 

I decided not to use Amazon Web Services for this project even though I was already familiar with AWS. Even on 
my own laptop, the training was fast enough. 


### Results

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

### Test a Model on New Images

I subsequently searched for a set of German traffic signs on the internet and tried how my model would perform in
classifying these images. 
 
The images are displayed below together with their class descriptions. 


I first preprocessed these images using the same colorspace conversions, histogram equalizations etc. as I performed on 
the original training set. Below are the preprocessed versions of the web images. 

![Preprocessed web images][preprocessed_web_images]


#### The Code to Produce Predictions

The pipeline to make the class prediction for the downloaded images can be found below. 
```python
def load_model_and_predict(features):
    """Load a model into a session again, and use it to produce predictions of image labels"""
    with tf.Session() as sess:
        restorer = tf.train.Saver()
        restorer.restore(sess, './lenet-2.ckpt')

        for ind, img in enumerate(features): 
            plt.imshow(img)
            plt.show()
            print(sess.run(tf.argmax(network['softmaxes'], 1), feed_dict={network['x']: [features[ind]]}))
```

I figured for a data set this small, it is easiest to again display the images next to their predicted labels. The output 
of running the function above for the model trained previously was as follows: 
 
![Downloaded images' prediction outputs][web_image_pred_output]

Here are the example results again in tabularized format (Note: the table is hard coded so it is not upated automatically 
upon subsequent runs):

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|

#### Discussion on Performance on Downloaded Images

FIXME

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### Evaluation of model confidence by using softmax probabilities 


```
FIXME: for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 
softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the 
"Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

```

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 




## Conclusion

