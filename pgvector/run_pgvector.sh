#!/bin/bash
docker pull -q ankane/pgvector
docker run -e POSTGRES_USER=joao \
           -e POSTGRES_PASSWORD=123 \
           -e POSTGRES_DB=mydatabase \
           --name my_postgres \
           -p 5432:5432 \
           -d ankane/pgvector
export PGPASSWORD='123'
sleep 5 
psql -h localhost -U joao -d mydatabase -p 5432 -c "CREATE EXTENSION vector;"

echo "pgvector container started and vector extension created."