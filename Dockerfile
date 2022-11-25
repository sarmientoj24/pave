FROM nvcr.io/nvidia/pytorch:21.10-py3

# Install linux packages
ENV TZ=Asia/Manila
RUN usermod -u 1000 www-data
RUN usermod -G staff www-data
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && \
    apt-get install -y \
        zip \
        htop \
        screen \
        libgl1-mesa-glx \
        git \
        ffmpeg \
        libsm6

COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip uninstall -y torch torchvision torchtext Pillow
RUN pip install --no-cache -r requirements.txt
RUN pip install git+https://github.com/juanmcasillas/gopro2gpx.git#egg=gopro2gpx

# RUN git clone https://github.com/sarmientoj24/pave.git /app/pave
COPY . /app/pave
WORKDIR /app/pave
# RUN pip install git+https://github.com/sarmientoj24/pave.git#egg=pave
RUN python setup.py develop
RUN mkdir data
RUN pip install gdown==4.5.4
RUN gdown https://drive.google.com/uc?id=1CR5PKZdX_xc-iVIhxAv4jG0nReQelMec