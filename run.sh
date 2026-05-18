cd /home/why/dbagent
/home/why/dbagent/pg_base/bin/pg_ctl -D /home/why/dbagent/pg_base/data stop -m immediate
/home/why/dbagent/pg_base/bin/pg_ctl -D /home/why/dbagent/pg_base/data start
/home/why/dbagent/pg_base/bin/psql -d postgres
