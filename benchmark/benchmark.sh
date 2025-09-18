#!/usr/bin/env bash
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

endpoint=$1
# Concurrent users
users=${2:-5}
# Duration in seconds
duration=${3:-60}
# Number of lines for each prompt (approx 85 tokens per line)
prompt_lines=${4:-18}
# Output tokens
tokens=${5:-250}

suffix=$(date +%Y%m%d%H%M%S)-${users}-users-${duration}-s

locust --headless  \
       --host ${endpoint} \
       -f benchmark/locust_client.py \
       --only-summary \
       --csv ${endpoint}-${suffix}.csv \
       --u ${users} \
       --run-time ${duration} \
       --prompt-file ${SCRIPT_DIR}/alice.txt \
       --average-prompt-lines ${prompt_lines} \
       --average-output-tokens ${tokens}
python ${SCRIPT_DIR}/benchmark_summary.py \
       --prefix ${endpoint}- \
       --summary_file ${endpoint}_summary.csv
