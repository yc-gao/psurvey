#!/bin/sh

if [ -n "${USER}" -a -n "${PASSWORD}" ]; then
    useradd -m -s /bin/zsh "${USER}"
    echo "${USER}:${PASSWORD}" | chpasswd
fi

exec "$@"

