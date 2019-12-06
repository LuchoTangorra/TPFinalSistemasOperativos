FROM tensorflow/tensorflow:nightly-gpu-py3

ADD Breast_Cancer_(CNN).py /
ADD X.npy /
ADD Y.npy /

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN pip3 install numpy
RUN pip3 install sklearn
RUN pip3 install wandb
RUN wandb login 83af0133d03ec9b087428803786afee30ea92958

CMD [ "python3", "./Breast_Cancer_(CNN).py" ]