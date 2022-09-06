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
make
make install
echo "[INFO] Finished"
cd ../..

# build pyrenderer
cd pyrenderer
echo "[INFO] Building pyrenderer"
bash compile-library-server.sh
echo "[INFO] Finished"
