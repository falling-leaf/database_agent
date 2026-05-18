#!/bin/bash

# 本地部署版第三方依赖下载与编译脚本

LOG_FILE="$HOME/pgdl/log.txt"
SCRIPT_PATH=$(cd "$(dirname "$0")" && pwd)
THIRD_PARTY_DIR="$HOME/pgdl/third_party"

function log() {
    local level=$1
    shift
    local message=$*
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[${timestamp}] [${level}] ${message}"
    echo "[${timestamp}] [${level}] ${message}" >> "${LOG_FILE}"
}

function loginfo() { log "INFO" "$*"; }
function logerror() { log "ERROR" "$*"; }

mkdir -p "$THIRD_PARTY_DIR"

# -------- 下载并编译 sentencepiece --------
loginfo "Cloning sentencepiece..."
git clone https://gitee.com/mirrors/sentencepiece "$THIRD_PARTY_DIR/sentencepiece"
if [[ $? -ne 0 ]]; then logerror "git clone sentencepiece error!"; exit 1; fi
loginfo "Clone sentencepiece success."

echo "add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)" >> "$THIRD_PARTY_DIR/sentencepiece/src/CMakeLists.txt"

cd "$THIRD_PARTY_DIR/sentencepiece"
mkdir -p build && cd build
cmake -DSPM_ENABLE_TCMALLOC=OFF ..
if [[ $? -ne 0 ]]; then logerror "generate sentencepiece makefile error!"; exit 1; fi
loginfo "Generate sentencepiece makefile success."

make -j$(nproc)
if [[ $? -ne 0 ]]; then logerror "build sentencepiece error!"; exit 1; fi
loginfo "Build sentencepiece success."

sudo make install
if [[ $? -ne 0 ]]; then logerror "install sentencepiece error!"; exit 1; fi
loginfo "Install sentencepiece success."

sudo mkdir -p /usr/local/share/SentencePiece
sudo mv "$SCRIPT_PATH/config/SentencePieceConfig.cmake" /usr/local/share/SentencePiece/

# -------- 下载并解压 libtorch --------
loginfo "Downloading libtorch..."
wget -P "$THIRD_PARTY_DIR" https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.0.0%2Bcpu.zip --no-check-certificate
if [[ $? -ne 0 ]]; then logerror "download libtorch error!"; exit 1; fi
loginfo "Download libtorch success."

unzip -d "$THIRD_PARTY_DIR" "$THIRD_PARTY_DIR"/*.zip
if [[ $? -ne 0 ]]; then logerror "unzip libtorch error!"; exit 1; fi
loginfo "Unzip libtorch success."
rm -f "$THIRD_PARTY_DIR"/*.zip

# -------- 下载并编译 OpenCV --------
loginfo "Downloading OpenCV..."
wget -P "$THIRD_PARTY_DIR" https://codeload.github.com/opencv/opencv/zip/refs/tags/3.4.16 --no-check-certificate
if [[ $? -ne 0 ]]; then logerror "download opencv error!"; exit 1; fi
loginfo "Download OpenCV success."

unzip -d "$THIRD_PARTY_DIR" "$THIRD_PARTY_DIR"/3.4.16
if [[ $? -ne 0 ]]; then logerror "unzip opencv error!"; exit 1; fi
loginfo "Unzip OpenCV success."
rm -f "$THIRD_PARTY_DIR"/3.4.16

cd "$THIRD_PARTY_DIR/opencv-3.4.16"
mkdir -p build && cd build
cmake ..
if [[ $? -ne 0 ]]; then logerror "generate opencv makefile error!"; exit 1; fi
loginfo "Generate OpenCV makefile success."

make -j$(nproc)
if [[ $? -ne 0 ]]; then logerror "build opencv error!"; exit 1; fi
loginfo "Build OpenCV success."

sudo make install
if [[ $? -ne 0 ]]; then logerror "install opencv error!"; exit 1; fi
loginfo "Install OpenCV success."

# -------- 下载并解压 onnxruntime --------
loginfo "Downloading onnxruntime..."
wget -P "$THIRD_PARTY_DIR" https://github.com/microsoft/onnxruntime/releases/download/v1.18.1/onnxruntime-linux-x64-1.18.1.tgz --no-check-certificate
if [[ $? -ne 0 ]]; then logerror "download onnxruntime error!"; exit 1; fi
loginfo "Download onnxruntime success."

tar -xvzf "$THIRD_PARTY_DIR/onnxruntime-linux-x64-1.18.1.tgz" -C "$THIRD_PARTY_DIR"
if [[ $? -ne 0 ]]; then logerror "unzip onnxruntime error!"; exit 1; fi
loginfo "Unzip onnxruntime success."

sudo mv "$SCRIPT_PATH/config/onnxruntimeConfig.cmake" "$THIRD_PARTY_DIR/onnxruntime-linux-x64-1.18.1/"

cd "$SCRIPT_PATH"
loginfo "All dependencies installed successfully."
