cd /home/why/dbagent/pgdl/third_party/postgresql-13.22/
make distclean
./configure --prefix=/home/why/dbagent/pg_base
make -j$(nproc)
make install
/home/why/dbagent/pg_base/bin/initdb -D /home/why/dbagent/pg_base/data
