## Copyright [2026] [Yijun Liu, Soochow University]
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

"""快速验证 requirements.txt 里的所有依赖能否正常导入。

用法:
  conda run -n ML python scripts/check_imports.py
  conda run -n ML python scripts/check_imports.py --requirements requirements.txt
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


def _import_name(dist_name: str) -> str:
    """把 PyPI 包名转成 import 名（多数一致，少数有映射）。"""
    mapping = {
        "pyyaml": "yaml",
        "python-dotenv": "dotenv",
        "python-dateutil": "dateutil",
        "rouge-score": "rouge_score",
        "scikit-learn": "sklearn",
        "pillow": "PIL",
        "pydantic-core": "pydantic_core",
        "pydantic_core": "pydantic_core",
        "pip-chill": "pip_chill",
        "bitsandbytes": "bitsandbytes",
        "huggingface-hub": "huggingface_hub",
        "huggingface_hub": "huggingface_hub",
        "hf-xet": "hf_xet",
        "safetensors": "safetensors",
        "sentencepiece": "sentencepiece",
        "typing-extensions": "typing_extensions",
        "typing_extensions": "typing_extensions",
        "typing-inspection": "typing_inspection",
        "typing_inspection": "typing_inspection",
        "async-timeout": "async_timeout",
        "async_timeout": "async_timeout",
        "nvidia-cublas-cu12": None,
        "nvidia-cuda-cupti-cu12": None,
        "nvidia-cuda-nvrtc-cu12": None,
        "nvidia-cuda-runtime-cu12": None,
        "nvidia-cudnn-cu12": None,
        "nvidia-cufft-cu12": None,
        "nvidia-curand-cu12": None,
        "nvidia-cusolver-cu12": None,
        "nvidia-cusparse-cu12": None,
        "nvidia-nccl-cu12": None,
        "nvidia-nvjitlink-cu12": None,
        "nvidia-nvtx-cu12": None,
        "triton": "triton",
        "setuptools": "setuptools",
        "wheel": "wheel",
        "pip": "pip",
        "absl-py": "absl",
        "markdown-it-py": "markdown_it",
        "torchvision": "torchvision",
        "torch": "torch",
        "tokenizers": "tokenizers",
        "datasets": "datasets",
        "transformers": "transformers",
        "accelerate": "accelerate",
        "peft": "peft",
        "trl": "trl",
        "openai": "openai",
        "requests": "requests",
        "numpy": "numpy",
        "pandas": "pandas",
        "scipy": "scipy",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "tqdm": "tqdm",
        "pytest": "pytest",
        "rich": "rich",
        "pydantic": "pydantic",
        "jieba": "jieba",
        "nltk": "nltk",
        "timm": "timm",
        "evaluate": "evaluate",
        "statsmodels": "statsmodels",
        "patsy": "patsy",
        "joblib": "joblib",
        "networkx": "networkx",
        "sympy": "sympy",
        "mpmath": "mpmath",
        "regex": "regex",
        "six": "six",
        "packaging": "packaging",
        "filelock": "filelock",
        "fsspec": "fsspec",
        "click": "click",
        "colorama": "colorama",
        "platformdirs": "platformdirs",
        "pluggy": "pluggy",
        "iniconfig": "iniconfig",
        "tomli": "tomli",
        "jiter": "jiter",
        "jinja2": "jinja2",
        "markupsafe": "markupsafe",
        "pygments": "pygments",
        "mdurl": "mdurl",
        "urllib3": "urllib3",
        "charset-normalizer": "charset_normalizer",
        "certifi": "certifi",
        "idna": "idna",
        "aiohttp": "aiohttp",
        "aiosignal": "aiosignal",
        "aiohappyeyeballs": "aiohappyeyeballs",
        "attrs": "attrs",
        "frozenlist": "frozenlist",
        "multidict": "multidict",
        "yarl": "yarl",
        "propcache": "propcache",
        "anyio": "anyio",
        "h11": "h11",
        "httpcore": "httpcore",
        "httpx": "httpx",
        "sniffio": "sniffio",
        "distro": "distro",
        "psutil": "psutil",
        "xxhash": "xxhash",
        "pyarrow": "pyarrow",
        "dill": "dill",
        "multiprocess": "multiprocess",
        "contourpy": "contourpy",
        "cycler": "cycler",
        "fonttools": "fontTools",
        "kiwisolver": "kiwisolver",
        "pyparsing": "pyparsing",
        "pillow": "PIL",
        "pytz": "pytz",
        "tzdata": "tzdata",
        "python-dateutil": "dateutil",
        "threadpoolctl": "threadpoolctl",
        "shellingham": "shellingham",
        "typer": "typer",
        "typer-slim": "typer",
        "annotated-doc": "annotated_doc",
        "annotated_types": "annotated_types",
        "exceptiongroup": "exceptiongroup",
        "pip-chill": "pip_chill",
    }
    key = dist_name.lower()
    if key in mapping:
        return mapping[key]
    # 普通规则：连字符转下划线
    return key.replace("-", "_")


def main():
    ap = argparse.ArgumentParser(description="验证 requirements.txt 所有依赖能否 import")
    ap.add_argument("--requirements", default="requirements.txt")
    ap.add_argument("-q", "--quiet", action="store_true", help="只打印失败项")
    args = ap.parse_args()

    req_path = Path(args.requirements)
    if not req_path.exists():
        print(f"[ERROR] requirements 文件不存在: {req_path}", file=sys.stderr)
        sys.exit(2)

    ok, fail, skip = [], [], []
    with open(req_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # 取包名（去掉版本约束和注释）
            pkg = line.split("==")[0].split(">=")[0].split("<=")[0].split("~=")[0].split("!=")[0]
            pkg = pkg.split(";")[0].split("#")[0].strip()
            if not pkg:
                continue
            imp = _import_name(pkg)
            if imp is None:
                skip.append(pkg)
                if not args.quiet:
                    print(f"[SKIP] {pkg:<30} (CUDA/系统库, 不验证 import)")
                continue
            try:
                importlib.import_module(imp)
                ok.append(pkg)
                if not args.quiet:
                    print(f"[OK]   {pkg:<30} -> import {imp}")
            except Exception as e:
                fail.append((pkg, imp, str(e)))
                print(f"[FAIL] {pkg:<30} -> import {imp}: {e}")

    print("\n=== 汇总 ===")
    print(f"  通过: {len(ok)}")
    print(f"  跳过: {len(skip)} (CUDA/系统库)")
    print(f"  失败: {len(fail)}")
    if fail:
        print("\n失败明细:")
        for pkg, imp, err in fail:
            print(f"  {pkg} (import {imp}): {err}")
        sys.exit(1)
    print("\n全部依赖 import 检查通过 ✓")


if __name__ == "__main__":
    main()
