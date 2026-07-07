# 犯罪意图识别框架 — ML 训练环境镜像
#
# 设计目标：在任意租来的 GPU 服务器上完美复刻本地 ML 环境
# 基础镜像：CUDA 12.4.1 + cuDNN 9（匹配 torch 2.5.1+cu124 自带的 CUDA 运行时）
#
# 用法：
#   docker build -t paper-ml:latest .
#   docker run --gpus all -it --rm \
#     -v $(pwd):/workspace -w /workspace \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     paper-ml:latest bash
#
# 验证环境：
#   docker run --gpus all --rm paper-ml:latest python /workspace/scripts/check_imports.py
#   docker run --gpus all --rm paper-ml:latest python -c "import torch; print(torch.cuda.is_available())"
#
# 镜像不含代码，代码在服务器上 git clone 后挂载进容器（/workspace）
# HF 缓存挂载到宿主机 ~/.cache/huggingface，避免重复下载模型

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

# 清华源（国内服务器加速；海外服务器把下面两行注释掉，或换成 https://pypi.org/simple）
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
 && pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 系统依赖：Python 3.10 + 常用构建工具
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        curl \
        ca-certificates \
        git \
        vim \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3.10-dev \
        python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# 升级 pip
RUN python -m pip install --upgrade pip

# 安装 Python 依赖（单独一层，requirements.txt 改动才重建）
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# 验证关键依赖能 import + CUDA 可用
RUN python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
 && python -c "import bitsandbytes as bnb; print('bitsandbytes', bnb.__version__)" \
 && python -c "import transformers, peft, trl, datasets, accelerate; print('transformers', transformers.__version__, 'peft', peft.__version__, 'trl', trl.__version__)" \
 && python -c "import matplotlib; matplotlib.use('Agg'); print('matplotlib OK')"

WORKDIR /workspace

CMD ["bash"]
