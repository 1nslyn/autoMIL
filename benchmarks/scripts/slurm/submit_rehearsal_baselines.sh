#!/bin/bash
# Native five-fold baselines for a rehearsal set (a cell-root directory beside
# the final grid, see runtime-rehearsal.roster.json). One job on one
# 3g.40gb slice of an H100 with two workers on it (a baseline training is
# feature-I/O bound: a tenth of a GPU's compute, ~16 GB of VRAM and ~50 GB
# of RAM per worker; BL_WORKERS_PER_GPU overrides the packing); each
# worker runs `campaign_stage.py run-baseline` for its share of the set's
# unregistered cells in roster order. Idempotent: registered cells are
# skipped. Nothing is mirrored to the export root: a rehearsal never enters
# the final grid.
#
# Usage, from the campaign checkout root, as the member who owns the set:
#   sbatch --account=def-jma-ab benchmarks/scripts/slurm/submit_rehearsal_baselines.sh runtime-rehearsal
#
#SBATCH --job-name=rehearsal_baselines
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --mem=96G
#SBATCH --output=logs/rehearsal_baselines_%j.out
#SBATCH --error=logs/rehearsal_baselines_%j.err

set -uo pipefail
RUNTIME_NAME="${1:?usage: submit_rehearsal_baselines.sh <runtime-name>}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:?submit from the campaign checkout root}"
CAMPAIGN_DIR="$PROJECT_DIR/benchmarks/campaigns/preprint_130"
RUNTIME="$CAMPAIGN_DIR/$RUNTIME_NAME"; ROSTER="$CAMPAIGN_DIR/$RUNTIME_NAME.roster.json"
cd "$PROJECT_DIR" || exit 1
[ -d "$RUNTIME" ] || { echo "ERROR: $RUNTIME is not materialized"; exit 1; }
[ -f "$ROSTER" ] || { echo "ERROR: $ROSTER missing (a rehearsal set needs its own roster)"; exit 1; }
umask 007
module load cuda/12.2 2>/dev/null || true
set -a; source benchmarks/.env; set +a
export UV_FROZEN=1 UV_NO_SYNC=1
N_GPUS="${SLURM_GPUS_ON_NODE:-1}"
N_WORKERS=$((N_GPUS * ${BL_WORKERS_PER_GPU:-2}))
LOG_DIR="$PROJECT_DIR/logs/baseline_cells/$RUNTIME_NAME"; mkdir -p "$LOG_DIR"

# Unregistered cells of the set, longest predicted first (5-fold hours from
# the final grid's ledger are unknown here, so the order is the roster's).
PENDING=$(uv run --frozen --no-sync --package autobench python - "$ROSTER" "$RUNTIME" <<'PYEOF'
import json, sys
from pathlib import Path
roster = json.loads(Path(sys.argv[1]).read_text()); runtime = Path(sys.argv[2])
for cell in roster["cell_ids"]:
    state_path = runtime / cell / "campaign_state.json"
    if not state_path.is_file():
        sys.exit(f"{cell}: not materialized under {runtime}")
    if json.loads(state_path.read_text()).get("baseline") is None:
        print(cell)
PYEOF
) || { echo "ERROR: $PENDING"; exit 1; }
echo "set $RUNTIME_NAME | $(echo "$PENDING" | grep -c .) unregistered cells | $N_WORKERS workers on $N_GPUS GPU(s) | $(hostname) | $(date)"
[ -n "$PENDING" ] || { echo "nothing to do"; exit 0; }

worker() {  # worker-id
    local id="$1" gpu=$(($1 % N_GPUS)) i=0 cell rc=0
    while IFS= read -r cell; do
        [ -n "$cell" ] || continue
        if [ $((i % N_WORKERS)) = "$id" ]; then
            echo "[gpu $gpu] $(date +%H:%M) run-baseline $cell"
            if uv run --frozen --no-sync --package autobench python benchmarks/scripts/campaign_stage.py run-baseline \
                    --cell-root "$RUNTIME/$cell" --gpu "$gpu" > "$LOG_DIR/$cell.log" 2>&1; then
                echo "[gpu $gpu] $(date +%H:%M) registered $cell"
            else
                echo "[gpu $gpu] $(date +%H:%M) FAILED $cell (see $LOG_DIR/$cell.log)"; rc=1
            fi
        fi
        i=$((i + 1))
    done <<< "$PENDING"
    return $rc
}
pids=(); for w in $(seq 0 $((N_WORKERS - 1))); do worker "$w" & pids+=($!); done
RC=0; for pid in "${pids[@]}"; do wait "$pid" || RC=1; done
chmod -R g+rwX "$RUNTIME" 2>/dev/null || true
echo "done rc=$RC $(date)"; exit $RC
