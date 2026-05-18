cd /home/why/dbagent
/home/why/dbagent/pg_base/bin/pg_ctl -D /home/why/dbagent/pg_base/data stop 2>/dev/null
cd /home/why/dbagent/pgdl
rm -rf build
mkdir build
cd /home/why/dbagent/pgdl/build
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
cmake -DCMAKE_PREFIX_PATH="/home/why/dbagent/pg_base/;/home/why/dbagent/pgdl/third_party/libtorch/;/home/why/dbagent/pgdl/third_party/onnxruntime-linux-x64-1.18.1/" ..
make -j$(nproc)
echo "morphingdbwhy" | sudo -S make install DESTDIR=/home/why/dbagent/pg_base
cp -f /home/why/dbagent/pg_base/home/why/dbagent/pg_base/lib/postgresql/pgdl.so /home/why/dbagent/pg_base/lib/postgresql/
cp -rf /home/why/dbagent/pg_base/home/why/dbagent/pg_base/lib/postgresql/pgdl /home/why/dbagent/pg_base/lib/postgresql/
cp -f /home/why/dbagent/pg_base/home/why/dbagent/pg_base/share/postgresql/extension/* /home/why/dbagent/pg_base/share/postgresql/extension/
