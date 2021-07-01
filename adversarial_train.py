from pyimagesearch.utils import get_class_idx
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import decode_predictions
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import argparse
import imutils
import cv2

def preprocess_image(image):
    # swap color channels, preprocess the image, and add in a batch
    # dimension
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocess_input(image)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    # return the preprocessed image
    return image

# load image from disk and make a clone for annotation
print("[INFO] loading image...")
image = cv2.imread(args["image"])
output = image.copy()
# preprocess the input image
output = imutils.resize(output, width=400)
preprocessedImage = preprocess_image(image)

# load the pre-trained ResNet50 model
print("[INFO] loading pre-trained ResNet50 model...")
model = ResNet50(weights="imagenet")
# make predictions on the input image and parse the top-3 predictions
print("[INFO] making predictions...")
predictions = model.predict(preprocessedImage)
predictions = decode_predictions(predictions, top=3)[0]

def clip_eps(tensor, eps):
    # clip the values of the tensor to a given range and return it
    return tf.clip_by_value(tensor, clip_value_min=-eps, clip_value_max=eps)


def generate_adversaries(model, baseImage, delta, classIdx, steps=50):
    # iterate over the number of steps
    for step in range(0, steps):
        # record our gradients
        with tf.GradientTape() as tape:
            # explicitly indicate that our perturbation vector should
            # be tracked for gradient updates
            tape.watch(delta)

            # add our perturbation vector to the base image and
            # preprocess the resulting image
            adversary = preprocess_input(baseImage + delta)
            # run this newly constructed image tensor through our
            # model and calculate the loss with respect to the
            # *original* class index
            predictions = model(adversary, training=False)
            loss = -keras.losses.SparseCategoricalCrossentropy()(tf.convert_to_tensor([classIdx]),
                predictions)
            # check to see if we are logging the loss value, and if
            # so, display it to our terminal
            if step % 5 == 0:
                print("step: {}, loss: {}...".format(step, loss.numpy()))
        # calculate the gradients of loss with respect to the
        # perturbation vector
        gradients = tape.gradient(loss, delta)
        # update the weights, clip the perturbation vector, and
        # update its value
        optimizer.apply_gradients([(gradients, delta)])
        delta.assign_add(clip_eps(delta, eps=EPS))
    # return the perturbation vector
    return delta


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
    help="path to original input image")
ap.add_argument("-o", "--output", required=True,
    help="path to output adversarial image")
ap.add_argument("-c", "--class-idx", type=int, required=True,
    help="ImageNet class ID of the predicted label")
args = vars(ap.parse_args())


# define the epsilon and learning rate constants
EPS = 2 / 255.0
LR = 0.1
# load the input image from disk and preprocess it
print("[INFO] loading image...")
image = cv2.imread(args["input"])
image = preprocess_image(image)

# load the pre-trained ResNet50 model for running inference
print("[INFO] loading pre-trained ResNet50 model...")
model = ResNet50(weights="imagenet")
# initialize optimizer and loss function
optimizer = Adam(learning_rate=LR)
sccLoss = SparseCategoricalCrossentropy()


# create a tensor based off the input image and initialize the
# perturbation vector (we will update this vector via training)
baseImage = tf.constant(image, dtype=tf.float32)
delta = tf.Variable(tf.zeros_like(baseImage), trainable=True)
# generate the perturbation vector to create an adversarial example
print("[INFO] generating perturbation...")
deltaUpdated = generate_adversaries(model, baseImage, delta, args["class_idx"])
# create the adversarial example, swap color channels, and save the
# output image to disk
print("[INFO] creating adversarial example...")
adverImage = (baseImage + deltaUpdated).numpy().squeeze()
adverImage = np.clip(adverImage, 0, 255).astype("uint8")
adverImage = cv2.cvtColor(adverImage, cv2.COLOR_RGB2BGR)
cv2.imwrite(args["output"], adverImage)






















