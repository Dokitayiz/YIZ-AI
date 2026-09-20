#!/bin/sh
set -eu
mc alias set local http://minio:9000 minio minio-password
mc mb --ignore-existing local/yiz-ai
