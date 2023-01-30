import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


def image_show_matplotlib(image):
    fig = plt.figure()
    plt.imshow(image, cmap='gray', interpolation='none')
    plt.title("Image")
    plt.xticks([])
    plt.yticks([])
    fig.show()

def predict(image):
    model = load_model('attempt.h5')
    #print(model.summary())
    config = model.get_config() # Returns pretty much every information about your model
    print("Input: ", config["layers"][0]["config"]["batch_input_shape"]) # returns a tuple of width, height and channels
    prediction = model.predict(image)

    return prediction, prediction.argmax()    
    
def prediction_show_matplotlib(cells):
    for i in range(len(cells)):
        cells[i] = 255 - cells[i]
    cells = cells 
    model = load_model('attempt.h5')
    pred = model.predict(cells)
    fig, axis = plt.subplots(4, 4, figsize=(12, 14))
    for i, ax in enumerate(axis.flat):
        ax.imshow(cells[i], cmap='binary')
        ax.set(title = f"Predicted Number is {pred[i].argmax()}")
    plt.show()

def convert_dtype(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img