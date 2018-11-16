# Emotional Face Classification

By Danny Lumian, PhD
November, 2018

# Motivation

Understanding emotions is crucial for navigating our complex, social world.

Faces are rich sources of emotional and non-verbal emotional information. 

Many of us excel at rapidly interpreting and aggreating such information.

However, sometimes it may be useful to augment or quantify such information. 

This project is aimed are providing such information for images and (eventually) videos. 

While emotion space can be conceptualized in several ways, one useful model is to break emotions into discrete categories.

This model will examine 7 'basic' emotion categories: 

    0:'Angry'
    1:'Disgust'
    2:'Fear'
    3:'Happy'
    4:'Sad'
    5:'Surprise'
    6:'Neutral'

Example Emotional Expressions : ![Expressions image](images/example_imgs.png "Examples of Emotional Expressions")

# Data: FERS2013

32,298 images across 7 categories

|    | Label    |   # train |   # bal train |   # test |
|---:|:---------|----------:|--------------:|---------:|
|  0 | Angry    |      4462 |           492 |      491 |
|  1 | Disgust  |       492 |           492 |       55 |
|  2 | Fear     |      4593 |           492 |      528 |
|  3 | Happy    |      8110 |           492 |      879 |
|  4 | Sad      |      5483 |           492 |      594 |
|  5 | Surprise |      3586 |           492 |      416 |
|  6 | Neutral  |      5572 |           492 |      626 |


# Data Exploration and Reduction

## PCA

Average faces by category: 

![PCA example faces](images/pca_images.png "PCA example faces")

Example faces with [1, 5, 10, 20] components:

![PCA example faces](images/pca_images_comparison.png "PCA example faces from components")

## NMF 

Example faces with [1, 5, 10, 20] components:

![NMF example faces](images/nmf_images_comparison.png "NMF example faces")


# Models

## Model Comparison

| Model           |   Balanced |   Train Log Loss |   Test Log Loss |   Train Accuracy |   Test Accuracy |
|:----------------|-----------:|-----------------:|----------------:|-----------------:|----------------:|
| MNB             |          1 |           25.997 |          26.357 |            0.24  |           0.233 |
| MNB             |          0 |           26.383 |          26.49  |            0.23  |           0.228 |
| Random_forest   |          1 |            0.368 |           6.981 |            0.994 |           0.225 |
| Random_forest   |          0 |            0.329 |           5.065 |            0.991 |           0.391 |
| CNN Categorical |          1 |           17.063 |          24.963 |            0.466 |           0.242 |
| CNN Continuous  |          1 |           19.464 |          23.281 |            0.411 |           0.305 |
| CNN Categorical |          0 |           33.277 |          33.287 |            0.027 |           0.026 |
| CNN Continuous  |          0 |           30.434 |          30.401 |            0.069 |           0.069 |


## Multinomial Naive Bayes

1. Vanilla (no parameters specified)
1. Good baseline model
1. 2-D images projected as 1-D arrays

### MNB

![MNB confusion matrix](images/MNB_not_balanced.png "MNB confusion matrix")

### MNB-Balanced

![MNB confusion matrix](images/MNB_balanced.png "MNB confusion matrix w/ balanced data")


## Random Forest

1. 100 trees
1. 2-D images projected as 1-D arrays
1. No pruning

### RF

![RF confusion matrix](images/Random_forest_not_balanced.png "Random Forest")

### RF-Balanced

![RF confusion matrix](images/Random_forest_balanced.png "Random Forest confusion matrix w/ balanced data")

## Convolutional Neural Network (CNN)

1. Compared continuous (ints) and categorical (OHE) target arrays

    + categorical_crossentropy (categorical)
    + sparse_categorical_crossentropy (continuous)

1. Batch size = 128
1. Epochs = 10
1. Steps_per_epoch = 50 
1. Used same model architecture for all 

```python
    nb_filters = 48
    kernel_size = (3, 3)
    pool_size = (2, 2)

    model = Sequential()
    # 2 convolutional layers followed by a pooling layer followed by dropout
    model.add(Convolution2D(nb_filters, kernel_size,
                            padding='valid',
                            input_shape=input_size))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))
    # transition to an mlp
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_categories))
    model.add(Activation('softmax'))
    return model
```

### Continuous CNN

![CNN continuous confusion matrix](images/Continuous_CNN.png "Cont CNN confusion matrix")

### Continuous CNN-Balanced

![CNN continuous confusion matrix](images/Continuous_CNN_bal.png "Cont CNN balanced confusion matrix")

### Categorical CNN

![CNN categorical confusion matrix](images/Categorical_CNN.png "Cat CNN confusion matrix")

### Categorical CNN-Balanced

![CNN categorical confusion matrix](images/Categorical_CNN_bal.png "Cat CNN balanced confusion matrix")

# Conclusions

1. Best model (to date) is Random Forest (with all data) with approx 40% accuracy 
1. Imbalanced data can cause problems
1. Important to check impact of parameters (i.e., class_weights)

# Future Directions

1. Improve model performance
1. Figure out weighting to utilize full training dataset
1. Image parsing to feed in new images, detect faces and categorize

