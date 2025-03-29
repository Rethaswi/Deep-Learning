# Deep-Learning

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: TEMURA RETHASWI

*INTERN ID*: CT04WM103

*DOMAIN*: FRONT END DEVELOPMENT

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTHOSH

##This Python script implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images from the CIFAR-10 dataset, a widely used benchmark dataset for computer vision tasks. The CIFAR-10 dataset consists of 60,000 color images of size 32x32 pixels, divided into 10 categories such as airplanes, cars, birds, and cats. The script begins by importing the necessary libraries, including TensorFlow for deep learning, Matplotlib for visualizing training progress, and Keras for defining and training the CNN model. The dataset is then loaded and split into training and testing sets, followed by normalization, where pixel values are scaled to a range between 0 and 1 to improve training efficiency. The CNN architecture is defined using Keras' Sequential API, consisting of three convolutional layers with ReLU activation functions, each followed by a max-pooling layer to reduce spatial dimensions while preserving essential features. The final layers include a flattening layer to convert the 2D feature maps into a 1D array, a fully connected dense layer with 128 neurons, and an output layer with 10 neurons using the softmax activation function to classify images into one of the 10 categories. The model is compiled with the Adam optimizer for efficient weight updates, sparse categorical cross-entropy loss for handling categorical labels, and accuracy as the evaluation metric. The training process runs for 10 epochs, using the training set for learning and the test set for validation, allowing the model to generalize better. After training, the model is evaluated on the test set, and the final test accuracy is printed to assess performance. To visualize the training process, the script plots the accuracy and validation accuracy across epochs, helping users analyze improvements and detect potential overfitting. This script is applicable in real-world scenarios such as object recognition, automated surveillance, and self-driving cars, where image classification plays a crucial role. Researchers and engineers can extend this model by adding data augmentation, experimenting with deeper architectures, or fine-tuning pre-trained models for improved accuracy. Additionally, this implementation serves as an excellent educational tool for those learning about CNNs and deep learning in image processing. By running the script, users can gain insights into how CNNs extract hierarchical features from images, enhancing their understanding of modern AI-driven image classification systems. This model can also be adapted for more complex datasets like ImageNet by scaling up the architecture and leveraging GPUs or TPUs for faster training. Overall, the script provides a solid foundation for developing and experimenting with deep learning models in computer vision applications.
