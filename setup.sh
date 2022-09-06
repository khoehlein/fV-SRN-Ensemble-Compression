cd external

# build tthresh
cd tthresh
echo "[INFO] Building TThresh"
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8
echo "[INFO] Finished"
cd ../..

# build sz3
cd sz3
echo "[INFO] Building SZ3"
sz3_dir=$(pwd)
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=$sz3_dir ..
make -j8
make install
echo "[INFO] Finished"
cd ../..

# build pyrenderer
cd pyrenderer
echo "[INFO] Building pyrenderer"
mkdir build
cd build
cmake -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.1 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.1/bin/nvcc -DTORCH_PATH=~/anaconda3/envs/python38torch18/lib/python3.8/site-packages/torch -DRENDERER_BUILD_GUI=OFF -DRENDERER_BUILD_TESTS=OFF -DRENDERER_BUILD_CLI=OFF -DRENDERER_BUILD_TESTS=OFF -DRENDERER_BUILD_OPENGL_SUPPORT=OFF ..
make -j8 VERBOSE=true
cd ..

echo "[INFO] Setup-Tools build"
python setup.py build
cp build/lib.linux-x86_64-3.8/pyrenderer.cpython-38-x86_64-linux-gnu.so bin/

echo "[INFO] Testing pyrenderer import"
cd bin
python -c "import torch; print(torch.__version__); import pyrenderer; print('[INFO] pyrenderer imported')"
