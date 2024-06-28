FROM python:3.11
WORKDIR /usr/src/app
COPY . .
RUN pip install torch torchvision
RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

RUN pip install  -r requirements.txt
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
CMD ["python", "app.py"]