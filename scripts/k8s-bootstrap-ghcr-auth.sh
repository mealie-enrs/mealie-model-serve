#!/usr/bin/env bash
# One-shot: create/update ghcr-pull and attach imagePullSecrets to app deployments.
# Run AFTER namespace + workloads exist (e.g. after first kubectl apply -k infra/k8s).
#
#   export GHCR_USERNAME=nidhish1
#   read -s GHCR_PAT && echo
#   ./scripts/k8s-bootstrap-ghcr-auth.sh
#
# Optional: KUBECTL="sudo k3s kubectl" NS=mms
set -euo pipefail
NS="${NS:-mms}"
KUBECTL="${KUBECTL:-kubectl}"

: "${GHCR_USERNAME:?set GHCR_USERNAME (GitHub user that owns the PAT)}"
: "${GHCR_PAT:?set GHCR_PAT (PAT with read:packages; use read -s GHCR_PAT)}"

$KUBECTL create secret docker-registry ghcr-pull \
  --docker-server=ghcr.io \
  --docker-username="$GHCR_USERNAME" \
  --docker-password="$GHCR_PAT" \
  -n "$NS" --dry-run=client -o yaml | $KUBECTL apply -f -

for d in mms-mlflow mms-model-serve; do
  $KUBECTL patch deployment "$d" -n "$NS" --type=strategic -p \
    '{"spec":{"template":{"spec":{"imagePullSecrets":[{"name":"ghcr-pull"}]}}}}'
done

echo "ghcr-pull + imagePullSecrets applied in ns/$NS. Roll out: kubectl rollout restart deployment/mms-mlflow -n $NS"
