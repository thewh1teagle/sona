#!/bin/bash
cd "$(dirname "$0")"

################################################################################
#
#       NEURLANG/IPA-WHISPER-MEDIUM MODEL
#
################################################################################
# PYTHON VERSION CHECK
################################################################################

py_version=$(python --version 2>&1)
pyfile_version=$(cat ../.python-version)
if [[ "$py_version" =~ Python\ ${pyfile_version}$ ]]; then
    echo "Your python is the correct version"
else
    echo "Your python ($py_version) is not the correct version ($pyfile_version), deleting ../.python-version"
    rm ../.python-version
fi

################################################################################
# GIT CLONE
################################################################################

# Make sure Vulkan dependencies are installed:
echo "Installing dependencies...."
sudo apt install -y build-essential cmake libvulkan-dev vulkan-tools \
                 vulkan-validationlayers libstdc++-12-dev libgomp1 glslc git-lfs

git lfs install
git clone https://github.com/ggml-org/ggml
git clone https://huggingface.co/neurlang/ipa-whisper-medium
git clone https://github.com/openai/whisper.git
git clone https://github.com/ggml-org/whisper.cpp.git



################################################################################
# GGML
################################################################################

cd ggml
git reset --hard v0.9.7
git clean -x -f -d

# install python dependencies in a virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# build the examples
mkdir build
cd build
cmake ..
cmake --build . --config Release -j 8

cd ../..

################################################################################
# WHISPER.CPP PREPARE convert-h5-to-ggml.py DEPS
################################################################################

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip uninstall -y tensorflow
pip install transformers safetensors numpy tensorflow-cpu

################################################################################
# WHISPER.CPP MAKE NEURLANG/IPA-WHISPER-MEDIUM MODEL
################################################################################

cd whisper.cpp
git reset --hard v1.8.3
git clean -x -f -d

python3 ./models/convert-h5-to-ggml.py \
    ../ipa-whisper-medium \
    ../whisper \
    .

mv ggml-model.bin models/ggml-base.en.bin

################################################################################
# WHISPER.CPP BUILD
################################################################################

# build the project
cmake -B build -S . \
    -DGGML_VULKAN=ON \
    -DCMAKE_BUILD_TYPE=Release  # Release build
cmake --build build -j --config Release

################################################################################
# WHISPER.CPP BACKTEST
################################################################################

# transcribe an audio file
./build/bin/whisper-cli -f samples/jfk.wav

cd ..


echo "Installing binaries...."
sudo cp whisper.cpp/build/src/libwhisper.so* /usr/local/lib/
sudo cp whisper.cpp/build/ggml/src/libggml*.so* /usr/local/lib/
sudo cp whisper.cpp/build/ggml/src/ggml-vulkan/libggml*.so* /usr/local/lib/
sudo ldconfig
