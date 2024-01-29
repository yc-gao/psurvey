#!/bin/sh

mkdir -p nginx/ssl
openssl genrsa -out nginx/ssl/private.key 2048
openssl req -new -x509 -days 3650 -key nginx/ssl/private.key -out nginx/ssl/server.crt -subj "/C=CN/ST=mykey/L=mykey/O=mykey/OU=mykey/CN=domain1/CN=domain2/CN=domain3"

