import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import cv2 as cv


def predict(image):
    from tensorflow.keras.models import load_model
    model = load_model('models/attempt.h5')
    #print(model.summary())
    config = model.get_config() # Returns pretty much every information about your model
    print("Input: ", config["layers"][0]["config"]["batch_input_shape"]) # returns a tuple of width, height and channels
    prediction = model.predict(image)

    return prediction, prediction.argmax() 

def predict_all(images, positions):
    from tensorflow.keras.models import load_model
    model = load_model('models/combined_dataset2.h5')
    predictions = model.predict(images)
    predict_list = [[predictions[i].argmax(), positions[i]] for i, pred in enumerate(predictions)]
    return predict_list    
    

def prediction_show_matplotlib(cells):
    from tensorflow.keras.models import load_model
    model = load_model('models/combined_dataset2.h5')
    pred = model.predict(cells)
    fig, axis = plt.subplots(4, 4, figsize=(12, 14))
    for i, ax in enumerate(axis.flat):
        ax.imshow(cells[i], cmap='gray')
        ax.set(title = f"Predicted Number is {pred[i].argmax()}")
    plt.show()