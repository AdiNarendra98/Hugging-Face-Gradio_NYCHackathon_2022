import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import requests

response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")

mobile_net = tf.keras.applications.MobileNetV2()

def classify_image_with_mobile_net(im):
    im = Image.fromarray(im.astype('uint8'), 'RGB')
    im = im.resize((224, 224))
    arr = np.array(im).reshape((-1, 224, 224, 3))
    arr = tf.keras.applications.mobilenet.preprocess_input(arr)
    prediction = mobile_net.predict(arr).flatten()
    return {labels[i]: float(prediction[i]) for i in range(1000)}
    
imagein = gr.inputs.Image()
label = gr.outputs.Label(num_top_classes=3)

examples = ['siamese.jpg','airplane.jpg','piano.jpg','pug.jpg','jeans.jpg']
article="<p style='text-align: center'>Made by Aditya Narendra with 🖤</p>"

gr.Interface(
    classify_image_with_mobile_net,
    imagein,
    label,
    title="Object Classifier",article=article,examples=examples,
    description="MobileNet trained on ImageNet with 1000 most common categories of Images ranging from animals to objects. Deployed on Hugging Faces using Gradio."
    ).launch()
