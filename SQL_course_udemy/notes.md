
# Installation of postgres on my linux:

sudo apt install curl

sudo curl https://www.pgadmin.org/static/packages_pgadmin_org.pub | sudo apt-key add

sudo sh -c 'echo "deb https://ftp.postgresql.org/pub/pgadmin/pgadmin4/apt/$(lsb_release -cs) pgadmin4 main" > /etc/apt/sources.list.d/pgadmin4.list && apt update'

sudo apt install pgadmin4
sudo apt install postgresql

sudo -u postgres psql template1
[sudo] password for vsevolod: 
could not change directory to "/home/vsevolod/Work/GIT/GitHub/afterglow_methods/cpp_model": Permission denied
psql (14.10 (Ubuntu 14.10-0ubuntu0.22.04.1))
Type "help" for help.

template1=# ALTER USER postgres with encrypted password MY4N;
ERROR:  syntax error at or near "MY4N"
LINE 1: ALTER USER postgres with encrypted password MY4N;
                                                    ^
template1=# ALTER USER postgres with encrypted password 'MY4N';
ALTER ROLE
template1=# exit

// HEre MY4N is my 4 numbers from my password


--- 

cd ~/Downloads/
sudo curl https://www.pgadmin.org/static/packages_pgadmin_org.pub | sudo apt-key add
sudo sh -c 'echo "deb https://ftp.postgresql.org/pub/pgadmin/pgadmin4/apt/$(lsb_release -cs) pgadmin4 main" > /etc/apt/sources.list.d/pgadmin4.list && apt update'
sudo apt install pgadmin4
sudo apt install postgresql
sudo -u postgres psql template1
sudo gedit /etc/postgresql/*/main/pg_hba.conf # cahnged one line to add postgre as 
    hostssl template1       postgres        192.168.122.1/24        scram-sha-256
sudo systemctl restart postgresql.service
sudo apt install postgresql-client
psql --host localhost --username postgres --password --dbname template1

---