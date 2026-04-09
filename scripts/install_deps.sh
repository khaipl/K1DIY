#!/bin/bash

echo "Mise à jour des paquets..."
sudo apt update

echo "Installation des dépendances..."
sudo apt install -y \
  cmake \
  ninja-build \
  libgtest-dev \
  libgoogle-glog-dev \
  libboost-dev \
  libeigen3-dev \
  liblua5.3-dev \
  graphviz \
  libgraphviz-dev \
  python3-pip \
  libcurl4-openssl-dev \
  libsdl2-dev \
  joystick \
  libspdlog-dev

echo "Installation terminée."
