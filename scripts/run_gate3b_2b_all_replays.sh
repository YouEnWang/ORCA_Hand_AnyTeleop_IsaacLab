#!/usr/bin/env bash

# ============================================================
# Gate 3B-2B Batch Offline Replay
#
# Purpose:
#   Run all Gate 3A trials through Gate 3B offline replay.
#
# Experiments:
#   5 motions
#   x 3 rounds
#   x 2 LPF settings
#   = 30 replay files
#
# LPF settings:
#   alpha = 0.2  -> canonical filtered output
#   alpha = 1.0  -> exact LPF bypass
#
# Logging:
#   All rounds belonging to the same motion are written into
#   one terminal log file.
#
# Output:
#   data/gate3b/replay/*.jsonl
#
# Logs:
#   results/gate3b/replay_logs/
#       static_open.txt
#       static_half.txt
#       static_pinch.txt
#       slow_open_close.txt
#       occlusion.txt
# ============================================================

set -euo pipefail


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT="/workspace/project"

REPLAY_SCRIPT="${PROJECT_ROOT}/scripts/replay_gate3b_retarget.py"

INPUT_DIR="${PROJECT_ROOT}/data/gate3a/raw"

OUTPUT_DIR="${PROJECT_ROOT}/data/gate3b/replay"

LOG_DIR="${PROJECT_ROOT}/results/gate3b/replay_logs"

CONFIG="${PROJECT_ROOT}/config/retargeting/orca_v2_right_vector_virtual_tip.yml"


# ============================================================
# Create output directories
# ============================================================

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"


# ============================================================
# Sanity checks
# ============================================================

if [[ ! -f "${REPLAY_SCRIPT}" ]]; then
    echo "[ERROR] Replay script not found:"
    echo "        ${REPLAY_SCRIPT}"
    exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
    echo "[ERROR] Retarget config not found:"
    echo "        ${CONFIG}"
    exit 1
fi


# ============================================================
# Motion list
# ============================================================

MOTIONS=(
    "static_open"
    "static_half"
    "static_pinch"
    "slow_open_close"
    "occlusion"
)


# ============================================================
# Initialize five log files
#
# Each new execution overwrites the previous batch logs.
# The JSONL replay outputs are also overwritten by Python.
# ============================================================

for motion in "${MOTIONS[@]}"; do

    log_file="${LOG_DIR}/${motion}.txt"

    : > "${log_file}"

    {
        echo "============================================================"
        echo "Gate 3B-2B Batch Replay Log"
        echo "============================================================"
        echo "Motion       : ${motion}"
        echo "Started at   : $(date --iso-8601=seconds)"
        echo "Project root : ${PROJECT_ROOT}"
        echo "Replay script: ${REPLAY_SCRIPT}"
        echo "Config       : ${CONFIG}"
        echo "============================================================"
        echo
    } >> "${log_file}"

done


# ============================================================
# Replay function
# ============================================================

run_replay() {

    local motion="$1"
    local round="$2"
    local alpha="$3"
    local alpha_tag="$4"

    local input_file
    local output_file
    local log_file

    input_file="${INPUT_DIR}/${motion}_r${round}.jsonl"

    output_file="${OUTPUT_DIR}/${motion}_r${round}_${alpha_tag}.jsonl"

    log_file="${LOG_DIR}/${motion}.txt"


    # --------------------------------------------------------
    # Verify source JSONL exists
    # --------------------------------------------------------

    if [[ ! -f "${input_file}" ]]; then

        {
            echo
            echo "============================================================"
            echo "[ERROR]"
            echo "Missing input file:"
            echo "${input_file}"
            echo "============================================================"
            echo
        } | tee -a "${log_file}"

        return 1
    fi


    # --------------------------------------------------------
    # Header for this individual replay
    # --------------------------------------------------------

    {
        echo
        echo
        echo "################################################################"
        echo "# Motion : ${motion}"
        echo "# Round  : R${round}"
        echo "# Alpha  : ${alpha}"
        echo "# Mode   : ${alpha_tag}"
        echo "# Input  : ${input_file}"
        echo "# Output : ${output_file}"
        echo "# Time   : $(date --iso-8601=seconds)"
        echo "################################################################"
        echo
    } | tee -a "${log_file}"


    # --------------------------------------------------------
    # Run replay
    #
    # 2>&1:
    #   Save both stdout and stderr.
    #
    # tee -a:
    #   Show output in terminal AND append to corresponding
    #   motion log.
    #
    # Because "set -o pipefail" is enabled, a Python failure
    # will still make this function fail.
    # --------------------------------------------------------

    if python "${REPLAY_SCRIPT}" \
        --input "${input_file}" \
        --output "${output_file}" \
        --config "${CONFIG}" \
        --alpha "${alpha}" \
        2>&1 | tee -a "${log_file}"
    then

        {
            echo
            echo "[PASS] ${motion} R${round} ${alpha_tag}"
            echo "Completed at: $(date --iso-8601=seconds)"
            echo
        } | tee -a "${log_file}"

    else

        {
            echo
            echo "[FAIL] ${motion} R${round} ${alpha_tag}"
            echo "Failed at: $(date --iso-8601=seconds)"
            echo
        } | tee -a "${log_file}"

        return 1
    fi
}


# ============================================================
# Batch execution
# ============================================================

echo
echo "============================================================"
echo "Gate 3B-2B Batch Replay"
echo "============================================================"
echo
echo "5 motions"
echo "x 3 rounds"
echo "x 2 LPF settings"
echo "= 30 replay runs"
echo


completed=0


for motion in "${MOTIONS[@]}"; do

    echo
    echo "============================================================"
    echo "Starting motion: ${motion}"
    echo "============================================================"

    for round in 1 2 3; do

        # ----------------------------------------------------
        # Canonical Gate 3A LPF
        # ----------------------------------------------------

        run_replay \
            "${motion}" \
            "${round}" \
            "0.2" \
            "alpha02"

        completed=$((completed + 1))


        # ----------------------------------------------------
        # Exact LPF bypass
        # ----------------------------------------------------

        run_replay \
            "${motion}" \
            "${round}" \
            "1.0" \
            "alpha10"

        completed=$((completed + 1))

    done


    # --------------------------------------------------------
    # Motion summary
    # --------------------------------------------------------

    {
        echo
        echo "============================================================"
        echo "[MOTION COMPLETE]"
        echo "Motion      : ${motion}"
        echo "Finished at : $(date --iso-8601=seconds)"
        echo "Expected replay files for this motion: 6"
        echo "============================================================"
        echo
    } | tee -a "${LOG_DIR}/${motion}.txt"

done


# ============================================================
# Final JSONL count
# ============================================================

json_count=$(
    find "${OUTPUT_DIR}" \
        -maxdepth 1 \
        -type f \
        -name "*.jsonl" \
        ! -name "*repeat*" \
        | wc -l
)


# ============================================================
# Final report
# ============================================================

echo
echo
echo "============================================================"
echo "Gate 3B-2B Batch Replay Finished"
echo "============================================================"
echo
echo "Replay commands completed : ${completed}"
echo "Formal JSONL files found  : ${json_count}"
echo
echo "Replay output directory:"
echo "  ${OUTPUT_DIR}"
echo
echo "Terminal log directory:"
echo "  ${LOG_DIR}"
echo


if [[ "${completed}" -eq 30 && "${json_count}" -eq 30 ]]; then

    echo "[PASS] All 30 formal Gate 3B-2B replay files were generated."

else

    echo "[CHECK]"
    echo "Expected:"
    echo "  completed = 30"
    echo "  JSONL     = 30"
    echo
    echo "Actual:"
    echo "  completed = ${completed}"
    echo "  JSONL     = ${json_count}"

fi


echo
echo "Generated terminal logs:"
echo

for motion in "${MOTIONS[@]}"; do
    echo "  ${LOG_DIR}/${motion}.txt"
done

echo
echo "============================================================"