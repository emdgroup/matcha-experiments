#!/usr/bin/env bash
# Generate the MATCHA autotune configs for every OpenADMET task group and architecture.
#
# Usage: bash configs/generate_autotune.sh [OUT_DIR]
#   OUT_DIR defaults to configs/autotune

set -euo pipefail

OUT_DIR="${1:-$(cd "$(dirname "$0")" && pwd)/autotune}"

GROUPS_LIST="caco2 clint physchem unbound"

# model short name : MATCHA architecture
MODELS="
attentivefp:AttentiveFPRegressor
chemprop:ChempropRegressor
cnn:CNNRegressor
e3gnn:E3GNNRegressor
gatedgcn:GatedGCNRegressor
gin:GINRegressor
gps:GPSRegressor
gps3d:GPS3DRegressor
gt:GTRegressor
gt3d:GT3DRegressor
mlp:MLPRegressor
rnn:RNNRegressor
roformer:RoFormerRegressor
snn:SNNRegressor
"

endpoints() {
  case "$1" in
    caco2)    echo "CACO2_Papp_AB_endpoint CACO2_ER_endpoint" ;;
    clint)    echo "HLM_CLint_endpoint MLM_CLint_endpoint" ;;
    physchem) echo "LogD_endpoint KSOL_endpoint" ;;
    unbound)  echo "MPPB_endpoint MBPB_endpoint MGMB_endpoint" ;;
    *) echo "Unknown task group: $1" >&2; exit 1 ;;
  esac
}

label_encoder() {
  local i=0
  for ep in $(endpoints "$1"); do
    printf '      %d:\n        task_label: "%s"\n' "$i" "$ep"
    i=$((i + 1))
  done
}

for group in $GROUPS_LIST; do
  mkdir -p "$OUT_DIR/$group"
  n_endpoints=$(endpoints "$group" | wc -w | tr -d ' ')

  for entry in $MODELS; do
    name="${entry%%:*}"
    arch="${entry##*:}"

    cat > "$OUT_DIR/$group/$name.yaml" <<EOF
dataset:
  path: "datasets/openadmet/train_$group.csv"
  smiles_key: "SMILES"
  label_key: "endpoint"
  operator_key: "OPERATOR"

split:
  method: "time"
  n_subset: 3
  method_params:
    key: "mol_id"
    split_size: 0.1

tuning:
  architecture_search:
    config: "default"
    budget: 100
  optimizer_search:
    config: "default"
    budget: 50

model:
  architecture: "$arch"
  params:
    loss_fn: "bounded-mse"
    num_endpoints: $n_endpoints
    num_epochs: 50
    batch_size: 64
    label_encoder_params:
$(label_encoder "$group")

output:
  optimum:
    path: "optima/$group"
    filename: "$name"
  mlflow:
    experiment_name: "autotune-$group-$name"
    log_dir: "mlflow"
    tags:
      architecture: "$arch"
      dataset: "openadmet-$group"
EOF
  done
done

echo "Configs written to $OUT_DIR"
