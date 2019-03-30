# **Behavioral Cloning** 

---
**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/final_architecture.png "Model Visualization"
[image2]: ./examples/left_center_right.png "Grayscaling"
[image3]: ./examples/normal_flipped.png "Recovery Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
or
```sh
python drive.py model2.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer.

#### 2. Attempts to reduce overfitting in the model

The model contains 1 BatchNormalization Layer then of each Conv2D layer in order to help us to reduce the training time and add some regularization. 

#### 3. Model parameter tuning

The model used an adam optimizer and ReduceLROnPlateau(to multiply the learn-rate by 0.1 if the val_loss is not decreasing then of 5 epochs), so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, adding to left measure 0.2 and substracting 0.2 to the right measure.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use 3 convolutional neural networks plus 2 dense layers but that was not sufficient, the model was performing so bad with the train data and valid data. Then I found the [nvidia paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) where I found a pretty good architecture which I used but adding some normalization steps.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that the nvidia model in my configuration had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that I added 1 BatchNormalization Layer then of each Conv2D layer in order to help us to reduce the training time and add some regularization. 

Finally I used from  ```keras.callbacks``` EarlyStopping(to stop the training then of some epochs without val_loss decreasement), ModelCheckpoint('to save the model each best epoch') and ReduceLROnPlateau(to multiply the learn-rate by 0.1 if the val_loss is not decreasing then of 5 epochs).

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)
(Here each Sequential Layer in each row in the next image represents a BatchNorm layer +  Conv2d layer)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. 
I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image2]

To augment the data sat, I also flipped images and angles thinking that this would help the model to generalize. For example, here is an image that has then been flipped:

![alt text][image3]

After the collection process, I had X number of data points. I then preprocessed this data by getting the correct color space and resizing the image to 200,66,3.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was more than 20 (knowing that I trained my model by 20 epochs and the val_loss was still decreasing) as evidenced in the 10th cell of the file model.ipynb.
