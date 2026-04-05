# Deploy mealie-model-serve on your Chameleon k3s VM (e.g. 192.5.86.170)

## Can we deploy there?

**Yes, if** the same pattern as your DMS box holds:

1. **k3s** is installed (your DMS Terraform `cloud-init` does this).
2. **`local-path` StorageClass** exists (k3s default).
3. You can **`kubectl apply`** from your laptop with kubeconfig, or run `kubectl` **over SSH** on the VM.

This repo’s **Docker Compose** stack is the same components; **Kubernetes** just schedules them as Pods + Services + PVCs.

**Terraform note:** Use **`mealie-model-serve/infra/terraform`** for a dedicated MMS VM (security group opens **22**, **30601**, **30608**; cloud-init installs Docker + k3s). Alternatively, reuse **`mealie/infra/terraform`** for the VM and open those NodePorts in Horizon or a separate rule set. Either way, **apply k8s manifests** (below) with kubectl; this Terraform stack does not install workloads.

We **cannot** verify `192.5.86.170` from the assistant’s environment; after SSH, run `kubectl get nodes` to confirm.

---

## 1) Build and push images

On your Mac (or CI), from **`mealie-model-serve/`** root:

```bash
export REG=ghcr.io/<your-org>   # or docker.io/...

docker build -f Dockerfile.mlflow -t ${REG}/mealie-model-serve-mlflow:latest .
docker build -f Dockerfile.serving --build-arg GIT_SHA=$(git rev-parse --short HEAD) \
  -t ${REG}/mealie-model-serve-api:latest .

docker push ${REG}/mealie-model-serve-mlflow:latest
docker push ${REG}/mealie-model-serve-api:latest
```

Edit **`infra/k8s/kustomization.yaml`** `images:` to match `${REG}` and tag.

On k3s without a registry mirror, you can **`docker save` | `ssh` | `sudo k3s ctr images import`** instead of push/pull.

---

## 2) Secrets

```bash
cp infra/k8s/secrets.example.yaml /tmp/mms-secrets.yaml
# edit passwords + MLFLOW_BACKEND_STORE_URI (password must match DATABASE_PASSWORD)
kubectl apply -f /tmp/mms-secrets.yaml
```

---

## 3) Apply manifests

```bash
export KUBECONFIG=~/.kube/config   # or scp from VM: /etc/rancher/k3s/k3s.yaml
kubectl apply -k infra/k8s/
kubectl -n mms rollout status deployment/mms-mlflow --timeout=300s
kubectl -n mms rollout status deployment/mms-model-serve --timeout=300s
```

---

## 4) Register a model (Job or laptop)

Same as local: run **`trainer/train.py`** or **`trainer/toy_random_onnx.py`** with:

- `MLFLOW_TRACKING_URI=http://<FLOATING_IP>:30601` (NodePort to MLflow), or port-forward:
  `kubectl -n mms port-forward svc/mlflow 5000:5000`

and MinIO/S3 env pointing at **`http://<FLOATING_IP>:<minio-nodeport>`** — MinIO is **ClusterIP** only in these manifests; for training from **outside** the cluster, either:

- add a **NodePort Service** for Minio (9000), or  
- `kubectl port-forward -n mms svc/minio 9020:9000` from your laptop while training.

---

## 5) URLs (replace `<IP>` with 192.5.86.170 if that is the node with NodePort routing)

| Service | URL |
|---------|-----|
| MLflow UI | `http://<IP>:30601` |
| Model API | `http://<IP>:30608` (e.g. `/metadata`, `/predict`) |

**Security group:** allow **TCP 30601** and **30608** (and any extra NodePorts you add) from where you browse/curl.

---

## 6) OpenStack / Terraform (optional)

If you want Terraform to open ports, add rules to the same security group as the DMS instance (example values — adjust names):

```hcl
resource "openstack_networking_secgroup_rule_v2" "mms_mlflow" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 30601
  port_range_max    = 30601
  remote_ip_prefix  = var.allowed_api_cidr
  security_group_id = openstack_networking_secgroup_v2.dms.id
}

resource "openstack_networking_secgroup_rule_v2" "mms_api" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 30608
  port_range_max    = 30608
  remote_ip_prefix  = var.allowed_api_cidr
  security_group_id = openstack_networking_secgroup_v2.dms.id
}
```

(Use your real `security_group_id` and CIDR.)

---

## Troubleshooting

- **`ImagePullBackOff`:** set `imagePullSecrets` for GHCR or import images on the node.  
- **`Pending` PVCs:** check `kubectl get sc` — need `local-path`.  
- **Model serve `503` on `/readyz`:** no registry model yet — run trainer against MLflow, then restart deployment or `POST /reload`.
