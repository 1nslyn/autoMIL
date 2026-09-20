#!/bin/bash
# Reproduction measurement for a rehearsal set: how far a re-run of each
# registered baseline lands from the registered value, per discovery fold
# (`campaign_stage.py run-baseline-reproduction --measure`; no verdict is
# recorded, and the discovery job supersedes a measurement-mode record with
# --force on its own). The summary at the end is the basis for the epsilon
# in reproduction_policy.json. One 3g.40gb slice of an H100, two workers.
#
# Submit from the campaign checkout root, after the set's baselines job:
#   sbatch --account=def-jma-ab --dependency=afterok:<baselines job> \
#       benchmarks/scripts/slurm/measure_reproduction.sh runtime-rehearsal
#
#SBATCH --job-name=measure_repro
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --mem=96G
#SBATCH --output=logs/measure_reproduction_%j.out
#SBATCH --error=logs/measure_reproduction_%j.err

set -uo pipefail
RUNTIME_NAME="${1:?usage: measure_reproduction.sh <runtime-name>}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:?submit from the campaign checkout root}"
cd "$PROJECT_DIR" || exit 1
CAMPAIGN="benchmarks/campaigns/preprint_130"
RUNTIME="$CAMPAIGN/$RUNTIME_NAME"
ROSTER="$CAMPAIGN/$RUNTIME_NAME.roster.json"
[ -d "$RUNTIME" ] || { echo "ERROR: $RUNTIME is not materialized"; exit 1; }
[ -f "$ROSTER" ] || { echo "ERROR: $ROSTER missing (a rehearsal set needs its own roster)"; exit 1; }
LOG_DIR="logs/baseline_cells/$RUNTIME_NAME"; mkdir -p "$LOG_DIR"
umask 007
module load cuda/12.2 2>/dev/null || true
set -a; source benchmarks/.env; set +a
export UV_FROZEN=1 UV_NO_SYNC=1
N_GPUS="${SLURM_GPUS_ON_NODE:-1}"
N_WORKERS=$((N_GPUS * ${BL_WORKERS_PER_GPU:-2}))
CELLS=$(python3 -c "import json; print('\n'.join(json.load(open('$ROSTER'))['cell_ids']))")
echo "measure | $(echo "$CELLS" | grep -c .) cells | $N_WORKERS workers on $N_GPUS GPU(s) | $(hostname) | $(git rev-parse --short HEAD) | $(date)"

worker() {  # worker-id
    local id="$1" gpu=$(($1 % N_GPUS)) i=0 cell rc=0
    while IFS= read -r cell; do
        [ -n "$cell" ] || continue
        if [ $((i % N_WORKERS)) = "$id" ]; then
            echo "[gpu $gpu] $(date +%H:%M) measure $cell"
            if uv run --frozen --no-sync --package autobench python benchmarks/scripts/campaign_stage.py \
                    run-baseline-reproduction --cell-root "$RUNTIME/$cell" --gpu "$gpu" --measure \
                    > "$LOG_DIR/$cell.measure.log" 2>&1; then
                echo "[gpu $gpu] $(date +%H:%M) measured $cell"
            else
                echo "[gpu $gpu] $(date +%H:%M) FAILED $cell (see $LOG_DIR/$cell.measure.log)"; rc=1
            fi
        fi
        i=$((i + 1))
    done <<< "$CELLS"
    return $rc
}
pids=(); for w in $(seq 0 $((N_WORKERS - 1))); do worker "$w" & pids+=($!); done
RC=0; for pid in "${pids[@]}"; do wait "$pid" || RC=1; done

python3 - "$RUNTIME" "$ROSTER" <<'PYEOF'
import json, sys
from pathlib import Path
runtime = Path(sys.argv[1]); cells = json.loads(Path(sys.argv[2]).read_text())["cell_ids"]
worst = 0.0
for cell in cells:
    block = json.loads((runtime / cell / "campaign_state.json").read_text()).get("baseline_reproduction") or {}
    deltas = [abs(float(row["delta"])) for row in block.get("folds", [])]
    matches = [row.get("prediction_hash_match") for row in block.get("folds", [])]
    peak = max(deltas) if deltas else None
    print(f"{cell}: mode={block.get('mode')} max_abs_delta={peak} hash_match={matches}")
    worst = max([worst] + deltas)
print(f"observed_max_abs_delta={worst:.6f}")
PYEOF
chmod -R g+rwX "$RUNTIME" 2>/dev/null || true
echo "done rc=$RC $(date)"; exit $RC
