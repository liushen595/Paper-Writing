# Docker 环境使用指南

把本地 ML 环境打包成 Docker 镜像，在任意租来的 GPU 服务器上完美复刻，无需手动装依赖。

## 一、本地构建镜像（一次性）

在项目根目录执行：

```bash
docker build -t paper-ml:latest .
```

构建约 15-30 分钟（取决于网络，主要耗时在 pip install torch + nvidia-cuda 库 ~3GB）。

验证镜像：

```bash
# 验证关键依赖 import + CUDA
docker run --gpus all --rm paper-ml:latest python -c "import torch; print(torch.cuda.is_available())"
# 完整 import 检查（需挂载代码）
docker run --gpus all --rm -v $(pwd):/workspace -w /workspace paper-ml:latest python scripts/check_imports.py
```

## 二、推送镜像到云端（租服务器前）

镜像约 8-10GB，推到 Docker Hub 或阿里云容器镜像服务（国内拉取快）。

### 方案 A：Docker Hub（海外服务器）
```bash
docker login
docker tag paper-ml:latest <your-username>/paper-ml:latest
docker push <your-username>/paper-ml:latest
```

### 方案 B：阿里云 ACR（国内服务器，推荐）
1. 登录 [阿里云容器镜像服务](https://cr.console.aliyun.com)，创建命名空间和镜像仓库
2. 按页面提示登录 + 推送：
```bash
docker login --username=<阿里云账号> registry-<地域>.aliyuncs.com
docker tag paper-ml:latest registry-<地域>.aliyuncs.com/<命名空间>/paper-ml:latest
docker push registry-<地域>.aliyuncs.com/<命名空间>/paper-ml:latest
```

### 方案 C：直接把镜像存成文件传输
```bash
docker save paper-ml:latest | gzip > paper-ml.tar.gz
# scp 到服务器后：
gunzip -c paper-ml.tar.gz | docker load
```
适合一次性使用，但文件 3-4GB，传输慢。

## 三、租到服务器后使用

### 1. 拉取镜像
```bash
# Docker Hub
docker pull <your-username>/paper-ml:latest
# 阿里云 ACR
docker pull registry-<地域>.aliyuncs.com/<命名空间>/paper-ml:latest
# 统一 tag 成 paper-ml
docker tag <完整镜像名> paper-ml:latest
```

### 2. clone 代码
```bash
git clone https://github.com/liushen595/Paper-Writing.git
cd Paper-Writing
```

### 3. 配置密钥
```bash
cp .env.example .env
vim .env  # 填入 GLM_API_KEY / AGNES_API_KEY 等
```

### 4. 启动容器
```bash
docker run --gpus all -it --rm \
  --name paper-train \
  --shm-size=16g \
  -v $(pwd):/workspace -w /workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.config/huggingface:/root/.config/huggingface \
  paper-ml:latest bash
```

参数说明：
- `--gpus all`：透传宿主机所有 GPU
- `--shm-size=16g`：Docker 默认 /dev/shm 只有 64MB，PyTorch DataLoader 多进程会 OOM，必须调大
- `-v $(pwd):/workspace`：把代码挂载进容器
- `-v ~/.cache/huggingface`：HF 模型/数据集缓存挂到宿主机，容器销毁后缓存保留，下次启动不用重新下
- `-v ~/.config/huggingface`：HF token（huggingface-cli login 写在这里）

### 5. 容器内首次登录 HF（WildChat 草垛 + 模型下载）
```bash
huggingface-cli login  # 粘贴 HF token
```
> 建议在宿主机 `~/.config/huggingface/token` 提前放好 token 文件，挂载后容器内免登录。

### 6. 跑训练
```bash
# 容器内
bash scripts/run_all.sh all
# 或后台跑（推荐 tmux）
tmux new -s train
bash scripts/run_all.sh all 2>&1 | tee outputs/run_all.log
# Ctrl+B D 脱离
```

## 四、常见问题

### Q1：服务器没装 Docker？
大多数 GPU 租赁平台（AutoDL、矩池云、阿里云 GN7 等）预装 Docker。若没有：
```bash
curl -fsSL https://get.docker.com | sh
systemctl start docker
# 装 NVIDIA Container Toolkit（让 docker --gpus all 生效）
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update && apt-get install -y nvidia-docker2
systemctl restart docker
```

### Q2：镜像太大拉取慢？
- 国内服务器用阿里云 ACR（同地域内网拉取，速度 50MB/s+）
- 或用方案 C 把镜像存成文件 scp 过去
- 或在服务器上直接 `docker build` （需服务器能访问清华 pip 源）

### Q3：训练中容器挂了怎么办？
代码和数据都在宿主机（挂载），容器挂了不影响。重新 `docker run` 即可。训练 checkpoint 在 `checkpoints/`（挂载到宿主机），可断点续训。

### Q4：怎么在容器外查看训练日志？
```bash
docker exec -it paper-train bash  # 进入正在跑的容器
# 或
docker logs -f paper-train
# 或挂载日志目录 -v $(pwd)/logs:/workspace/logs
```

### Q5：怎么改镜像里的依赖？
改 `requirements.txt` 后重新 `docker build -t paper-ml:latest .`。pip 层有缓存，只重建受影响部分。
