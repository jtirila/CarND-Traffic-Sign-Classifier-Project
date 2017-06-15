# **Traffic Sign Recognition** 

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

[web_images]: ./web_images.png "Web images"
[web_images_preprocessed]: ./web_images_preprocessed.png "Preprocessed web images"
[relative_amounts]: ./relative_amounts.png "Relative amounts of different traffic signs"
[prediction_80kmh]: ./prediction_80kmh.png "Prediciton 80 km / h"
[prediction_snow]: ./prediction_snow.png "Prediction snow"
[example_images]: ./example_images.png "Example images"
[example_unprocessed]: ./example_unprocessed.png "An unprocessed example image"
[example_preprocessed]: ./example_processed.png "A preprocessed example image"


## Dataset exploration

### Basic Stats and Visualizations

Here is a basic summary of training, validation and testing data sets, including the numbers of images per each of the
traffic sign categories:


```
     Class index |   # in train set |    # in test set | # in validation set 
-----------------|------------------|------------------|------------------
               0 |              180 |               60 |               30
               1 |             1980 |              720 |              240
               2 |             2010 |              750 |              240
               3 |             1260 |              450 |              150
               4 |             1770 |              660 |              210
               5 |             1650 |              630 |              210
               6 |              360 |              150 |               60
               7 |             1290 |              450 |              150
               8 |             1260 |              450 |              150
               9 |             1320 |              480 |              150
              10 |             1800 |              660 |              210
              11 |             1170 |              420 |              150
              12 |             1890 |              690 |              210
              13 |             1920 |              720 |              240
              14 |              690 |              270 |               90
              15 |              540 |              210 |               90
              16 |              360 |              150 |               60
              17 |              990 |              360 |              120
              18 |             1080 |              390 |              120
              19 |              180 |               60 |               30
              20 |              300 |               90 |               60
              21 |              270 |               90 |               60
              22 |              330 |              120 |               60
              23 |              450 |              150 |               60
              24 |              240 |               90 |               30
              25 |             1350 |              480 |              150
              26 |              540 |              180 |               60
              27 |              210 |               60 |               30
              28 |              480 |              150 |               60
              29 |              240 |               90 |               30
              30 |              390 |              150 |               60
              31 |              690 |              270 |               90
              32 |              210 |               60 |               30
              33 |              599 |              210 |               90
              34 |              360 |              120 |               60
              35 |             1080 |              390 |              120
              36 |              330 |              120 |               60
              37 |              180 |               60 |               30
              38 |             1860 |              690 |              210
              39 |              270 |               90 |               30
              40 |              300 |               90 |               60
              41 |              210 |               60 |               30
              42 |              210 |               90 |               30

```
| Category | Explanation | Amount in training set | Amount in validation set | Amount in test set |
| -------- | ----------- | ---------------------: | -----------------------: | -----------------: |


Here is the code that was used to produce the table above:


```python
orig_label_stats_train = {ind: list(y_train).count(ind) for ind in set(y_train)}
orig_label_stats_valid = {ind: list(y_valid).count(ind) for ind in set(y_valid)}
orig_label_stats_test = {ind: list(y_test).count(ind) for ind in set(y_test)}

print("{:>16} | {:>16} | {:>16} | {:>16} ".format('Class index', '# in train set', '# in test set', '# in validation set'))
print("-"*17 + "|" + "-" * 18 + "|" + "-" * 18 + "|" + "-" * 18 )
for label in range(43):
    print("{:>16} | {:>16} | {:>16} | {:>16}".format(label, orig_label_stats_train[label], orig_label_stats_test[label], orig_label_stats_valid[label]))
    
 ```
 
See below for the same info for training data in a visual format. 

![relative_amounts]

The figure below contains an illustration of 9 randomly selected images from the training set.

![example_images] 



## Image Color Normalization and augmentation 

I ended up feeding the images to the preprocessor in RGB format, and that is also the format used for the 
actual training. 

I experimented also with grayscaled images but ended up using the three color channels. 
Looking at the visualizations, it is obvious that the images vary greatly in luminosity and contrast. I hence looked 
for a method to enhance the contrast and also unify the image luminosity / brighness across examples. The method 
I ended up using is to perform histrogram normalization on the Y channel of the image, temporarily first converted 
into the YUV colorspace format for this procedure. The code to do this using OpenCV can be found below. 

```python
# Normalize contrast as per http://stackoverflow.com/a/38312281 """
img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

# equalize the histogram of the Y channel
channel = img_yuv[:, :, 0]
# plt.imshow(channel)
# plt.show()
img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

# convert the YUV image back to RGB format
color_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
```

The following two figures contain first an example of an unprocessed image, and subsequently the preprocessed version 
of the same image.
![An example of a raw input image][example_unprocessed]
![An example of a preprocessed image][example_preprocessed]

As another preprocessing step, I augment the data so that each class is approximately equally represented in the
training data set. This could be done by just copying some images over and over, but I figured it is better to introduce
"new" examples by performing some slight alterations of the images while copying. 

Here is the code that simultaneously evens out the numbers of examples per class and produces the slight rotations 
and scalings: 

```python
    orig_label_stats_train = {ind: list(train_labels).count(ind) for ind in set(train_labels)}
    orig_label_stats_valid = {ind: list(valid_labels).count(ind) for ind in set(valid_labels)}
    max_num_labels = max(orig_label_stats_train.values())

    for label, num in orig_label_stats_train.items():
        # Find all images with this label
        imgs = [img_label[0] for img_label in zip(train_features, train_labels) if img_label[1] == label]

        coeff = max_num_labels / orig_label_stats_train[label] - 1
        coeff_int = int(np.floor(coeff))
        original_length = len(imgs)
        if coeff_int > 0:
            imgs_repeated = np.array([img for img in imgs for _ in range(coeff_int)])
        else:
            imgs_repeated = np.array([])
        max_ind = int((coeff - coeff_int) * original_length)
        if max_ind > 0:
            if len(imgs_repeated) > 0:
                imgs_repeated = np.concatenate((imgs_repeated, imgs[:max_ind]))
            else:
                imgs_repeated = np.array(imgs[:max_ind])

        rotation_angles = [24 * (random() - 0.5) for _ in range(len(imgs_repeated))]
        scale_coeffs = [1 + 0.5 * (random() - 0.5) for _ in range(len(imgs_repeated))]

        for ind, img in enumerate(imgs_repeated):
            matr = cv2.getRotationMatrix2D((16, 16), rotation_angles[ind], scale_coeffs[ind])
            imgs_repeated[ind] = cv2.warpAffine(img, matr, (32, 32))

        if len(imgs_repeated) > 0:
            train_features = np.concatenate((train_features, imgs_repeated))
            train_labels = np.concatenate([train_labels, np.tile(label, len(imgs_repeated))])

```

With this augmentation technique, I ended up with a data set of about 84 000 examples. 

### Design and Test a Model Architecture

The model architecture is a sligthly modified version of the LeNet architecture. The LeNet implementation written
for previous phases of the curse seemed like a good starting point above all for the nice match of input dimension,
but also because of the good performance and relative simplicity of the architecture in comparison to some of the
arguably more advanced architectures. I ended up modifying LeNet by adding depth dimensions to the convolution layers
and adding a dropout layer. 

As the loss function I used the averaged cross entrypy function as described in the project instruction. 
I tried out a couple of different optimizers: `AdamOptimizer`, `RMSPropOptimizer`
and `MomentOptimizer`, and `AdamOptimizer´ seemed to yield the best results. 

As for batch_size (128), the weight initialization guassian parameters (mu = 0.0, sigma = 0.05), 
learning rate (0.0014) and number of epochs (18), I pretty much ended up experimenting around and getting a feel for 
what works and what does not. For many combinations of the parameters the performance of the model would 
be really bad and finding suitable ones took quite a bit of time. 

The model and training code is pretty similar to that in the instructions so I am not going to repeat it here. 

#### Dropout During Prediction Phase?

One phenomenon that requires some attention is that I observed varying results even after training the model and 
producing predictions. It turned out the dropout I was using was also active during prediction phase. 

To eliminate this effect, I initialized the keep_prob value as a TensorFlow placeholder, provide it to further 
processing in the feed_dict wherever applicable, and in prediction phase set it to 1.0. This way I was able 
to benefit from dropout during training but still obtain consistent predictions after the training phase. 

### Perfomance of the model

Here is a transcript of the final run of the training: 

```
2017-06-15 18:48:36.659990: EPOCH 1 ...
Train Accuracy = 0.788458, validation accuracy: 0.698337
2017-06-15 18:48:52.599361: EPOCH 2 ...
Train Accuracy = 0.921806, validation accuracy: 0.842043
2017-06-15 18:49:08.576955: EPOCH 3 ...
Train Accuracy = 0.956152, validation accuracy: 0.873159
2017-06-15 18:49:24.577733: EPOCH 4 ...
Train Accuracy = 0.971833, validation accuracy: 0.874822
2017-06-15 18:49:40.518170: EPOCH 5 ...
Train Accuracy = 0.976485, validation accuracy: 0.894695
2017-06-15 18:49:56.458764: EPOCH 6 ...
Train Accuracy = 0.984944, validation accuracy: 0.907838
2017-06-15 18:50:12.465146: EPOCH 7 ...
Train Accuracy = 0.982202, validation accuracy: 0.895170
2017-06-15 18:50:28.364046: EPOCH 8 ...
Train Accuracy = 0.986067, validation accuracy: 0.900713
2017-06-15 18:50:44.324132: EPOCH 9 ...
Train Accuracy = 0.984829, validation accuracy: 0.897466
2017-06-15 18:51:00.233405: EPOCH 10 ...
Train Accuracy = 0.992733, validation accuracy: 0.911006
2017-06-15 18:51:16.152955: EPOCH 11 ...
Train Accuracy = 0.992235, validation accuracy: 0.911481
2017-06-15 18:51:32.055862: EPOCH 12 ...
Train Accuracy = 0.988080, validation accuracy: 0.903088
2017-06-15 18:51:47.955031: EPOCH 13 ...
Train Accuracy = 0.988034, validation accuracy: 0.890974
2017-06-15 18:52:03.905439: EPOCH 14 ...
Train Accuracy = 0.992617, validation accuracy: 0.906176
2017-06-15 18:52:19.859192: EPOCH 15 ...
Train Accuracy = 0.986611, validation accuracy: 0.898496
2017-06-15 18:52:35.969826: EPOCH 16 ...
Train Accuracy = 0.992837, validation accuracy: 0.912352
2017-06-15 18:52:52.023942: EPOCH 17 ...
Train Accuracy = 0.986021, validation accuracy: 0.899525
2017-06-15 18:53:08.213401: EPOCH 18 ...
Train Accuracy = 0.992177, validation accuracy: 0.910530
```

As can be seen, the accuracy on the training set is even indicative of a a little overfitting. However, the 
validation accuracy is not really decresgin, and also results on the validation set are reasonable as seen below 
(my description of the result is wrong, this truly is the result of the accuracy operation on _validation_ data.

```
Accuracy on test data set: 0.9405895691609978
```

## Test a Model on New Images

I subsequently searched and downloaded some images from the internet. The collection is illustrated below. 

![Images downloaded from the internet][web_images]

Here the preprocessing has been applied to these images. In addition to luminosity equalization, these images needed 
to also be resized to 32x32 pixels to be processable by the LeNet architecture. 

[web_images_preprocessed]: ./web_images_preprocessed.png "Preprocessed web images"


### The Prediction Results for New Images

The notebook contains complete records of how the model performed with these images it had never seen before. 

To summarize briefly, one of the five images was misclassified, the 80 km /h speed limit sign was mistakenly 
labeled as a 50 km / h sign. The rest were correctly labeled in their respective classes. 

Here is yet another illustrations one of these images, together with an example from the same category in the training 
data. 

![Prediction for the snow warning][prediction_snow]

And here is the misclassified sign with an example of its predicted category: 


![The 80 km/h sign and its predicted counterpart][prediction_80kmh]


### Softmaxes

As a final step, I performed the softmax calculations as required. 

Here are the softmax values for the misclassified sign: 

```
Max_softmaxes:
TopKV2(values=array([[  9.99185622e-01,   3.91835987e-04,   2.00213355e-04,
          1.18437107e-04,   7.95269807e-05]], dtype=float32), indices=array([[2, 3, 7, 1, 5]], dtype=int32))

```

The correct label is 5, but the image was labeled 2. 

Here are the softmax values for the no entry sign (label 17): 

```
Max_softmaxes:
TopKV2(values=array([[  1.00000000e+00,   6.24160279e-09,   6.08771701e-25,
          2.35997533e-28,   5.99968284e-29]], dtype=float32), indices=array([[17, 14,  9, 10, 34]], dtype=int32))

```

One can see that the max softmax value dominates the others dramatically. This was systematically the case for the 
images easier to classify. 

## Discussion and Improvement Ideas

The project was rather hard in terms of parameter tuning. I am not sure if there is something off with my code, but I 
needed to iterate really really long to get things working properly. 

As for the preprocessing part, the project was very enlightening. In addition to the luminosity histogram equalization, I 
experimented with a range of techniques from OpenCV, including various kinds of blurring and adaptive contrast 
enhancement methods. However, due to the nature of the data (only 32x32 pixels), the other methods didn't provide 
as good results. 

Due to the great amount of work with the network parameter tuning, I was not able to perform as much of data augmentation 
as I had wished. Flipping the images, adding noise and distorting them in other ways may have been useful. 

Another technique I was thinking of was masking the input images outside an area of interest. Despite the differing 
shapes and sizes (triangle, circle, stop sign, rectangle, ...), I think setting the pixels at the corners to e.g. zero
could have improved the performance of the network by discarding unimportant information. 

A goal I had with the different smoothing techniques was to find something that would preserve uniform areas in an 
image and discard basically any local phenomena. I am not sure if this a common approach in a deep learning context, 
but maybe it would be useful to first recognize just the overall shape within the image and subsequently 
perform different kinds of further preprocesing outside and inside of the actual traffic sign area. 

