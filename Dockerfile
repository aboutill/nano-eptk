FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install base utilities
RUN apt-get update \
    && apt install -y software-properties-common \
    && apt update \
    && apt-get install -y --no-install-recommends \
    	build-essential wget git cmake cmake-curses-gui python3 python3-pip libtbb-dev \
    	libboost-all-dev libeigen3-dev zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
    	libssl-dev libreadline-dev libffi-dev zram-config fltk1.3-dev \ 
    	mesa-utils libglm-dev file dc pulseaudio libquadmath0 libgtk2.0-0 firefox libgomp1 \
    	g++ libfftw3-dev libtiff5-dev libpng-dev \
    	libqt5opengl5-dev libqt5svg5-dev libglx-mesa0 libgl1 python-is-python3 \
    	ninja-build qt6-base-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
    
# Install mrtrix
RUN git clone https://github.com/MRtrix3/mrtrix3.git /home/mrtrix3
RUN cd /home/mrtrix3 \
    && git checkout dev \
    && cmake -B build -G Ninja \
    && cmake --build build \
    && cmake --install build --prefix /usr/local/mrtrix3dev 
# Update path
ENV PATH="/usr/local/mrtrix3dev/bin:${PATH}"

# Install MIRTK
RUN git clone https://github.com/SVRTK/MIRTK.git /home/MIRTK
RUN cd /home/MIRTK \
    && mkdir build \
    && cd build/ \
    && cmake -D WITH_TBB="OFF" -D CMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-Wno-error=changes-meaning -Wno-changes-meaning" .. \
    && make -j
RUN cp /home/MIRTK/build/bin/mirtk /usr/local/bin/
# Update path
ENV PATH="/home/MIRTK/build/bin:/home/MIRTK/build/lib/tools:${PATH}"

# Install FSL
RUN wget https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/fslinstaller.py
RUN python ./fslinstaller.py -d /usr/local/fsl/
# Update path and FSL env
ENV FSLDIR=/usr/local/fsl
ENV FSLOUTPUTTYPE=NIFTI_GZ
ENV PATH="/usr/local/fsl/bin:${PATH}"

# Make the image runnable as an arbitrary non-root UID
ENV HOME=/home/user
RUN mkdir -p /home/user && chmod -R 777 /home/user
ENV TMPDIR=/home/user/tmp
RUN mkdir -p /home/user/tmp && chmod 777 /home/user/tmp
ENV MPLCONFIGDIR=/home/user/tmp/matplotlib

# Set a directory for the app
WORKDIR /usr/src/app

# Copy requirement
COPY requirements.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy files
COPY ./src ./src
COPY setup.py setup.py

# Install fcmr_reconstruction_app
RUN pip install -e .

# Expose port 8888 for Jupyter notebook server forwarding
EXPOSE 8888

