#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run as root: sudo bash scripts/bootstrap-gpu-node.sh"
  exit 1
fi

SSH_USER="${SUDO_USER:-cc}"

apt-get update
apt-get install -y --no-install-recommends curl jq gnupg docker.io
systemctl enable --now docker
usermod -aG docker "${SSH_USER}" || true

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y --no-install-recommends nvidia-container-toolkit nvidia-container-runtime

curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC='server --write-kubeconfig-mode 644' sh -

mkdir -p /var/lib/rancher/k3s/agent/etc/containerd
cat > /var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl <<'EOF'
{{ template "base" . }}

[plugins.'io.containerd.cri.v1.runtime'.containerd.runtimes.'nvidia']
  runtime_type = "io.containerd.runc.v2"

[plugins.'io.containerd.cri.v1.runtime'.containerd.runtimes.'nvidia'.options]
  BinaryName = "/usr/bin/nvidia-container-runtime"
  SystemdCgroup = false
EOF

systemctl restart k3s

mkdir -p "/home/${SSH_USER}/.kube"
cp /etc/rancher/k3s/k3s.yaml "/home/${SSH_USER}/.kube/config"
chown -R "${SSH_USER}:${SSH_USER}" "/home/${SSH_USER}/.kube"

for _ in $(seq 1 60); do
  if /usr/local/bin/k3s kubectl get nodes >/dev/null 2>&1; then
    break
  fi
  sleep 5
done

/usr/local/bin/k3s kubectl apply -f \
  https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.0/deployments/static/nvidia-device-plugin.yml
/usr/local/bin/k3s kubectl patch daemonset nvidia-device-plugin-daemonset \
  -n kube-system \
  --type merge \
  -p '{"spec":{"template":{"spec":{"runtimeClassName":"nvidia"}}}}'
/usr/local/bin/k3s kubectl rollout status daemonset/nvidia-device-plugin-daemonset -n kube-system --timeout=180s

echo "GPU node bootstrap complete."
echo "Next: apply mms-secrets, then deploy the chameleon-s3-gpu overlay."
