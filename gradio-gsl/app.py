import gradio as gr
import json
import numpy as np


from gsl_utils import single_process


def gsl(image):
    gsl_result = single_process(image)

    return gsl_result


examples = [
    ["demo.jpg"]
]

gr.Interface(
    fn=single_process,
    inputs=[gr.Image(type="pil")],
    outputs=[gr.Image(label='GSL', type="numpy"),],
    title="RB-IBDM Yolo Demo 🐞",
    examples=examples
).launch()
