#!/bin/bash
#
# Script to run the fine-tuning
#
# Copyright (C) 2025 Human Dataware Lab.
# Created by HDL members
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

log()
{
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# General
stage=1
stop_stage=4

# Training related
train_dataset_dir=./data/train
test_dataset_dir=./data/test
train_config=./configs/presets/config_dit_mel_seed_uvit_xlsr_tiny.yml
run_name=""

# Evaluation related
checkpoint=""
reference_audio_path=""
inference_opts=""

help_message=$(
    cat << EOF
Usage: $0 [options]

Options:
    --stage <stage>                Stage to start from (default: \"${stage}\")
    --stop_stage <stop_stage>      Stage to stop at (default: \"${stop_stage}\")
    --train_dataset_dir <dir>      Directory containing the training dataset (default: \"${train_dataset_dir}\")
    --test_dataset_dir <dir>       Directory containing the test dataset (default: \"${test_dataset_dir}\")
    --train_config <file>          Path to the training configuration file (default: \"${train_config}\")
    --run_name <name>              Name for the run (default: \"${run_name}\")
    --checkpoint <file>            Path to the checkpoint file for evaluation (default: \"${checkpoint}\")
    --reference_audio_path <path>  Path to the reference wav file for evaluation (default: \"${reference_audio_path}\")
    --inference_opts <opts>        Additional options for inference (default: \"${inference_opts}\")
EOF
)

log "$0 $*"
# shellcheck disable=SC1091
. utils/parse_options.sh

set -euo pipefail

if [ ! -d "${train_dataset_dir}" ]; then
    log "Train dataset directory ${train_dataset_dir} does not exist. Please provide a valid train dataset directory."
    exit 1
fi
if [ ! -d "${test_dataset_dir}" ]; then
    log "Test dataset directory ${test_dataset_dir} does not exist. Please provide a valid test dataset directory."
    exit 1
fi
if [ ! -f "${train_config}" ]; then
    log "Train config file ${train_config} does not exist. Please provide a valid train config file."
    exit 1
fi
if [ -z "${run_name}" ]; then
    run_name=$(date +%Y%m%d_%H%M%S)_$(basename "${train_config}" .yml)
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    log "Stage 1: Data validation"
    uv run python local/validate_dataset.py \
        --config "${train_config}" \
        "${train_dataset_dir}"
    log "Data validation for training dataset completed successfully."
    uv run python local/validate_dataset.py \
        --config "${train_config}" \
        "${test_dataset_dir}"
    log "Data validation for test dataset completed successfully."
fi

if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    log "Stage 2: Training"
    log "Training start. See logs in the run directory: runs/${run_name}"
    uv run python -m seed_vc.bin.train \
        --config "${train_config}" \
        --dataset-dir "${train_dataset_dir}" \
        --run-name "${run_name}"
    log "Training completed successfully."
fi

set +o pipefail
if [ -z "${checkpoint}" ]; then
    log "Checkpoint file is not provided. Using the latest checkpoint from the run directory."
    checkpoint=$(find "runs/${run_name}" -type f -name "*.pth" | sort | tail -n 1)
fi
if [ -z "${reference_audio_path}" ]; then
    log "Reference audio file is not provided. Using the first wav file from the train dataset."
    reference_audio_path=$(find "${train_dataset_dir}" -type f -name "*.wav" | head -n 1)
fi
output_dir="runs/${run_name}/inferences/$(basename "${checkpoint}" .pth)"
set -o pipefail

if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    log "Stage 3: Inference"
    if [ -z "${checkpoint}" ]; then
        log "No checkpoint found in runs/${run_name}. Please provide a valid checkpoint file."
        exit 1
    fi
    if [ -z "${reference_audio_path}" ]; then
        log "No wav file found in ${test_dataset_dir}. Please provide a valid reference audio file."
        exit 1
    fi
    # shellcheck disable=SC2086
    uv run python -m seed_vc.bin.inference \
        ${inference_opts} \
        --source "${test_dataset_dir}" \
        --target "${reference_audio_path}" \
        --output "${output_dir}" \
        --config "${train_config}" \
        --checkpoint "${checkpoint}"
    log "Inference completed successfully."
fi

if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ]; then
    log "Stage 4: Evaluation"
    log "Evaluating converted audio against the reference audio..."
    uv run python -m local.eval \
        --converted-dir "${output_dir}" \
        --reference "${reference_audio_path}" \
        --output-dir "${output_dir}/eval/converted"
    log "Evaluation of converted audio completed. Results saved to ${output_dir}/eval/converted"

    log "Evaluating original audio against the reference audio..."
    uv run python -m local.eval \
        --converted-dir "${test_dataset_dir}" \
        --reference "${reference_audio_path}" \
        --output-dir "${output_dir}/eval/original"
    log "Evaluation of original audio completed. Results saved to ${output_dir}/eval/original"
fi

log "All stages completed successfully (Elapsed time: ${SECONDS}s)."
