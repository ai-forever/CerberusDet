# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-04.html
FROM nvcr.io/nvidia/pytorch:23.04-py3

# Install linux packages
RUN apt update && apt install -y zip htop screen libgl1-mesa-glx libfreetype6-dev

# Install python dependencies
RUN python -m pip install --upgrade pip
RUN pip uninstall -y nvidia-tensorboard nvidia-tensorboard-plugin-dlprof

# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy contents
COPY . /usr/src/app
RUN pip install -e .

RUN pip install --no-cache coremltools onnx gsutil notebook
RUN pip install --no-cache opencv-python==4.8.0.74 Pillow==9.5.0

# Set environment variables
ENV HOME=/usr/src/app
