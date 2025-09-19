#!/usr/bin/env bash
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

endpoint=$1
# Concurrent users
users=${2:-5}
# Duration in seconds
duration=${3:-60}

locust --headless  \
       --host ${endpoint} \
       -f benchmark/locust_client.py \
       --only-summary \
       --csv ${endpoint}-${users}users-${duration}s.csv \
       --u ${users} \
       --run-time ${duration}
python ${SCRIPT_DIR}/benchmark_summary.py \
       --prefix ${endpoint}- \
       --summary_file ${endpoint}_summary.csv
