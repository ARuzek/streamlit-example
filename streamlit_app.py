from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import torch
print("PyTorch Version :{}".format(torch.__version__))
import torchvision
print("TorchVision Version :{}".format(torchvision.__version__))

import matplotlib
print("Matplotlib Version :{}".format(matplotlib.__version__))

import numpy as np
print("Numpy Version :{}".format(np.__version__))



"""
# Welcome to Streamlit! A

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""


with st.echo(code_location='below'):
    total_points = st.slider("Number of points in spiral", 1, 5000, 2000)
    num_turns = st.slider("Number of turns in spiral", 1, 100, 9)

    Point = namedtuple('Point', 'x y')
    data = []

    points_per_turn = total_points / num_turns

    for curr_point_num in range(total_points):
        curr_turn, i = divmod(curr_point_num, points_per_turn)
        angle = (curr_turn + 1) * 2 * math.pi * i / points_per_turn
        radius = curr_point_num / total_points
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        data.append(Point(x, y))

    st.altair_chart(alt.Chart(pd.DataFrame(data), height=500, width=500)
        .mark_circle(color='#0068c9', opacity=0.5)
        .encode(x='x:Q', y='y:Q'))

# Load Model
from torchvision.models import resnet50, ResNet50_Weights

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.eval();
preprocess_func = ResNet50_Weights.IMAGENET1K_V2.transforms()
preprocess_func
categories = np.array(ResNet50_Weights.IMAGENET1K_V2.meta["categories"])
len(categories), categories[:5]

# Make Prediction
from PIL import Image
shark = Image.open("shark.jpg")

shark.size
shark
#preprocess image
processed_img = preprocess_func(shark)
processed_img.shape
#predict
probs = model(processed_img.unsqueeze(0))
probs = probs.softmax(1)
probs = probs[0].detach().numpy()
#sort probs
prob = probs[probs.argsort()[-5:][::-1]]
idxs = probs.argsort()[-5:][::-1]
prob, idxs
categories[idxs]
