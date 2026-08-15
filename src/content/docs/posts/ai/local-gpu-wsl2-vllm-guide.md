---
title: "[환경 구축 기록] WSL2·Docker에서 로컬 RTX 4070 GPU로 LLM 실행하기"
description: "Windows PC의 RTX 4070을 WSL2·Docker에 연결하고, JupyterLab에서 Qwen 실습을 실행하며 겪은 설정·검증·트러블슈팅 기록"
---

:::note[환경 구축 기록]
Windows PC(RTX 4070 / VRAM 12GB)에서 WSL2, NVIDIA Container Toolkit, Docker를 연결하고 JupyterLab으로 Qwen 실습을 실행하기까지의 설정·검증·트러블슈팅 기록입니다.
:::

---

:::note[Quick Summary: 로컬 GPU 실습 아키텍처 한눈에 보기]
```
[ Windows 11 / 10 Host OS ] (최신 NVIDIA GeForce 드라이버 설치)
       │
       ▼ (WSL2 Direct GPU Pass-through)
[ WSL2 Ubuntu 22.04 LTS ] (Linux 서브시스템 환경)
       │
       ▼ (NVIDIA Container Toolkit / nvidia-ctk)
[ Docker Engine (GPU 지원) ]
       ├─────────────────────────────────────────┐
       ▼                                         ▼
[ JupyterLab 컨테이너 (Port 8888) ]     [ vLLM API 서버 컨테이너 (Port 8000) ]
(Python SDK / 벤치마크 테스트)         (로컬 실습: Qwen2.5-0.5B 추론)
```
:::

---

## 1. 개요 및 사전 준비사항

Windows 환경에서 vLLM이나 최신 딥러닝 서빙 프레임워크를 돌릴 때 가장 권장되는 아키텍처는 **WSL2(Windows Subsystem for Linux 2)** 기반 구축입니다. 

NVIDIA의 최신 Windows 드라이버는 WSL2 내부로 GPU를 직접 전달(Direct Pass-through)하므로, WSL2 내부에서 별도의 Linux용 GPU 드라이버를 중복 설치할 필요가 없습니다.

### 사전 준비 장비 및 스펙
- **OS**: Windows 11 또는 Windows 10 (Build 19044 이상)
- **GPU**: NVIDIA GeForce RTX 4070 (VRAM 12GB) / RTX 3000~4000 시리즈
- **권장 RAM**: 시스템 메모리 32GB 이상
- **디스크 공간**: 50GB 이상의 여유 공간 (HuggingFace 모델 및 Docker 이미지 저장용)

---

## 2. [1단계] Windows NVIDIA 드라이버 확인 및 WSL2 Ubuntu 설치

### ① Windows NVIDIA 드라이버 업데이트
NVIDIA 공식 홈페이지에서 최신 GeForce Game Ready 드라이버 또는 Studio 드라이버를 설치합니다.

### ② WSL2 및 Ubuntu 22.04 LTS 설치
PowerShell을 **관리자 권한**으로 실행하고 다음 명령을 입력합니다.

```powershell
# 1. WSL2 및 Ubuntu 22.04 설치
wsl --install -d Ubuntu-22.04

# 2. 설치 완료 후 Ubuntu 진입 (또는 윈도우 시작 메뉴에서 Ubuntu 실행)
wsl -d Ubuntu-22.04
```

### ③ WSL2 내부 GPU 인식 검증
Ubuntu 터미널 창에서 바로 `nvidia-smi`를 실행합니다.

```bash
nvidia-smi
```

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 555.58.02              Driver Version: 555.58.02    CUDA Version: 12.5     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Mode  | Disp.A / Shared Mem    | Memory-Usage         |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4070       On   |     On /  Shared       |   1100MiB / 12282MiB |
+-----------------------------------------+------------------------+----------------------+
```
위와 같이 **RTX 4070 (12GB VRAM)** 명칭과 CUDA 버전이 정상 출력되면 WSL2 GPU 연동 기본 준비가 완료된 것입니다.

---

## 3. [2단계] WSL2 내 Docker CE 및 NVIDIA Container Toolkit 설치

Docker 컨테이너 안에서 GPU를 활용하려면 **NVIDIA Container Toolkit**이 필요합니다.

WSL2 Ubuntu 터미널에서 다음 명령을 순서대로 실행합니다.

```bash
# 1. 기### Docker GPU 통과 테스트
Docker 안에서 GPU가 정상 인식되는지 테스트 컨테이너를 실행해 봅니다.

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

```
[ RTX 4070 실제 성공 검증 로그 ]
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4070        On  |   00000000:01:00.0  On |                  N/A |
|  0%   56C    P0             28W /  200W |    2411MiB /  12282MiB |      0%      Default |
+-----------------------------------------+------------------------+----------------------+
```
컨테이너 내부에서 `NVIDIA GeForce RTX 4070` 표가 정상 출력되면 WSL2 + Docker GPU 연동이 성공한 것입니다.


---

## 4. [3단계] 통합 JupyterLab + PyTorch GPU 실습 컨테이너 구축

PyTorch와 CUDA 환경이 갖춰진 컨테이너를 띄우고 JupyterLab을 연동하여 코드를 작성하고 테스트할 수 있는 개발 환경을 구축합니다.

:::note[소요 시간 및 이미지 크기 안내]
`pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel` 이미지에는 CUDA 12.1 개발 툴킷 및 cuDNN 개발 라이브러리가 풀 스택으로 포함되어 있어 **다운로드 압축 파일이 약 7GB, 디스크 압축 해제 용량은 무려 16.6GB**에 달합니다. 
네트워크 속도에 따라 **최초 풀링(Pulling) 시 약 3분~10분 정도 소요**되므로 다운로드가 완수될 때까지 느긋하게 기다려 주세요.
:::

### ① JupyterLab 컨테이너 실행
```bash
# 1. 작업용 로컬 디렉토리 생성
mkdir -p ~/llm-workspace

# 2. PyTorch GPU 컨테이너 실행 (WSL2 GPU 라이브러리 경로 LD_LIBRARY_PATH 추가)
docker run -d \
  --name llm-lab \
  --gpus all \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -p 8888:8888 \
  -p 8000:8000 \
  -v ~/llm-workspace:/workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --ipc=host \
  pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel \
  tail -f /dev/null


```

```
[ RTX 4070 환경 이미지 다운로드 완수 및 용량 검증 로그 ]
Status: Downloaded newer image for pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel
0a0f92f30dd607d2bbf199f8f5d1844ba092c048ba01fb669c72adfb6c7fc77a

$ docker image ls | grep pytorch
pytorch/pytorch  2.1.2-cuda12.1-cudnn8-devel  ecb3f786af6c  2 years ago  16.6GB
```



### ② 컨테이너 내부 환경 설정 및 JupyterLab 라이브러리 설치

:::note[vLLM 및 관련 라이브러리 설치 소요 시간 안내]
`pip install vllm` 실행 시 Triton, Transformers, Ray, FlashAttention 등 C++/CUDA 익스텐션 및 무거운 딥러닝 종속 패키지들이 함께 설치됩니다. 
**최초 패키지 설치에 약 7분~8분 정도 소요**될 수 있으므로 터미널에서 작업이 완료될 때까지 기다려 주세요.
:::

```bash
# 1. 컨테이너 내부 접속
docker exec -it llm-lab bash

# 2. vLLM 및 주피터 관련 필수 패키지 설치 (약 7분~8분 소요)
pip install vllm jupyterlab matplotlib pandas requests

# 3. JupyterLab 서버 실행
jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser --NotebookApp.token=''
```

### 주피터 접속�함되어 있어 **다운로드 압축 파일이 약 7GB, 디스크 압축 해제 용량은 무려 16.6GB**에 달합니다. 
네트워크 속도에 따라 **최초 풀링(Pulling) 시 약 3분~10분 정도 소요**되므로 다운로드가 완수될 때까지 느긋하게 기다려 주세요.
:::

### ① JupyterLab 컨테이너 실행
```bash
# 1. 작업용 로컬 디렉토리 생성
mkdir -p ~/llm-workspace

# 2. PyTorch GPU 컨테이너 실행 (WSL2 GPU 라이브러리 경로 LD_LIBRARY_PATH 추가)
docker run -d \
  --name llm-lab \
  --gpus all \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -p 8888:8888 \
  -p 8000:8000 \
  -v ~/llm-workspace:/workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --ipc=host \
  pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel \
  tail -f /dev/null


```

```
[ RTX 4070 환경 이미지 다운로드 완수 및 용량 검증 로그 ]
Status: Downloaded newer image for pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel
0a0f92f30dd607d2bbf199f8f5d1844ba092c048ba01fb669c72adfb6c7fc77a

$ docker image ls | grep pytorch
pytorch/pytorch  2.1.2-cuda12.1-cudnn8-devel  ecb3f786af6c  2 years ago  16.6GB
```



### ② 컨테이너 내부 환경 설정 및 JupyterLab 라이브러리 설치

:::note[⏳ vLLM 및 관련 라이브러리 설치 소요 시간 안내]
`pip install vllm` 실행 시 Triton, Transformers, Ray, FlashAttention 등 C++/CUDA 익스텐션 및 무거운 딥러닝 종속 패키지들이 함께 설치됩니다. 
**최초 패키지 설치에 약 7분~8분 정도 소요**될 수 있으므로 터미널에서 작업이 완료될 때까지 기다려 주세요.
:::

```bash
# 1. 컨테이너 내부 접속
docker exec -it llm-lab bash

# 2. vLLM 및 주피터 관련 필수 패키지 설치 (약 7분~8분 소요)
pip install vllm jupyterlab matplotlib pandas requests

# 3. JupyterLab 서버 실행
jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser --NotebookApp.token=''
```


### 🌐 주피터 접속
윈도우 웹 브라우저(Chrome/Edge)를 열고 `http://localhost:8888`로 접속하면 로컬 RTX 4070 GPU와 연결된 **JupyterLab 대시보드**가 열립니다.

```
[ JupyterLab 성공 구동 및 노트북 커널 시작 검증 로그 ]
[I ServerApp] Serving notebooks from local directory: /workspace
[I ServerApp] Jupyter Server 2.20.0 is running at: http://0.0.0.0:8888/lab
[I ServerApp] 302 GET / (@172.17.0.1) 0.45ms
[I ServerApp] Creating new notebook in /workspace
[I ServerApp] Kernel started: 51ceacac-4e56-4f29-98d7-8ea2aabc382b
[I ServerApp] Connecting to kernel 51ceacac-4e56-4f29-98d7-8ea2aabc382b.
```


---

## 5. [4단계] vLLM 모델 서빙 및 Python SDK 실습 테스트

이제 주피터 노트북이나 컨테이너 터미널에서 vLLM을 활용해 모델을 직접 서빙하고 추론을 돌려봅니다.

### ① RTX 4070(12GB VRAM) 맞춤형 vLLM 서빙 실행 (CLI)

로컬 검증용 기본 모델은 원본 실습과 동일한 `Qwen/Qwen2.5-0.5B`로 설정합니다. 이 모델은 앞선 Transformers 실습에서 사용한 모델과 같으며, 12GB VRAM에서 vLLM 엔진의 로딩과 텍스트 완성 흐름을 확인하기에 충분합니다.

:::caution[WSL2 컨테이너에서 `UVA is not available` 오류가 날 때]
일부 최신 vLLM 환경은 `Using V2 Model Runner` 경로에서 UVA(통합 가상 주소) 버퍼를 초기화합니다. WSL2·Docker 조합에서 이 초기화가 실패하면 `RuntimeError: UVA is not available`가 발생합니다. 모델 다운로드나 VRAM 부족 문제는 아닙니다. 이 경우 V1 엔진을 끄는 것이 아니라, 아래처럼 `VLLM_USE_V2_MODEL_RUNNER=0`으로 V2 Model Runner만 비활성화합니다.
:::

:::caution[7B 모델을 그대로 올릴 수 없는 이유]
`Qwen2.5-7B-Instruct`를 `bfloat16` 또는 `float16`으로 올리면 가중치만 약 14GB 이상이 필요합니다. RTX 4070의 VRAM은 12GB이므로 KV Cache 공간까지 포함하면 서버가 준비 완료 전에 종료될 수 있습니다. `--gpu-memory-utilization`은 KV Cache에 쓸 VRAM 비율이지 모델을 압축하는 옵션이 아닙니다. 7B를 사용하려면 양자화된 모델(AWQ/GPTQ 등)이나 더 큰 VRAM이 필요합니다.
:::

```bash
# OpenAI 호환 API 서빙 서버 실행 (RTX 4070 로컬 실습용)
# WSL2에서 UVA 오류가 발생한 환경은 V2 Model Runner를 끈다.
VLLM_USE_V2_MODEL_RUNNER=0 python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B \
  --port 8000 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 2048 \
  --dtype float16
```

터미널에 `Uvicorn running on http://0.0.0.0:8000`이 표시되면 API 서버가 준비된 것입니다. 아래 JupyterLab 실습을 마칠 때까지 이 터미널은 종료하지 않습니다.

### ② JupyterLab에서 vLLM 서버로 원본 생성 실습

원본 실습의 모델·영문 프롬프트·샘플링 설정을 유지하되, `LLM(...).generate(...)`로 엔진을 새로 만들지 않고 이미 실행한 vLLM 서버에 요청합니다. `Qwen2.5-0.5B`는 Base 모델이므로 대화용 `/v1/chat/completions`가 아니라 텍스트 완성용 `/v1/completions`를 사용합니다.

:::note[실습 중 확인한 환경 이슈와 반영]
vLLM 0.26.0에서 V2 Model Runner 경로는 `UVA is not available` 오류를 발생시켰고, `VLLM_USE_V2_MODEL_RUNNER=0`으로 우회했다. 또한 RTX 4070(12GB)에서 7B BF16 모델은 가중치 용량 때문에 API 준비 완료 전 종료될 수 있어, 원본 실습과 같은 0.5B 모델로 진행했다.
:::

```python
import time
import requests

# 현재 vLLM 서버가 제공하는 모델명을 읽어 요청 모델과 불일치하는 404를 피한다.
models = requests.get("http://127.0.0.1:8000/v1/models", timeout=10)
models.raise_for_status()
model_name = models.json()["data"][0]["id"]
print("서빙 중인 모델:", model_name)

prompt = """You are an expert AI historian writing a
detailed chapter for a book titled "The Evolution of
Human-AI Collaboration." Write in a formal tone,
with rich detail and examples in each era."""

start = time.perf_counter()
response = requests.post(
    "http://127.0.0.1:8000/v1/completions",
    json={
        "model": model_name,
        "prompt": prompt,
        "temperature": 0.8,
        "top_p": 0.95,
        "max_tokens": 128,
    },
    timeout=120,
)
response.raise_for_status()
elapsed = time.perf_counter() - start

generated_text = response.json()["choices"][0]["text"]
print("Generated text:", generated_text)
print(f"생성 시간: {elapsed:.2f}초")
```

`LLM(...).generate(...)`에 전달하던 `prompt`, `temperature`, `top_p`, `max_tokens`를 OpenAI 호환 API 요청 본문으로 옮긴 것이다. 생성 엔진은 서버에서 한 번만 유지하므로 GPU 메모리를 중복 점유하지 않는다.

#### 실행 결과 (RTX 4070 로컬 환경)

```text
서빙 중인 모델: Qwen/Qwen2.5-0.5B
Generated text: In the eleventh century, people thought that robots could
make themselves smarter than humans. What did they think about? ...
생성 시간: 0.42초
```

요청부터 128토큰 생성까지 약 0.42초가 걸려, 서버를 경유한 vLLM 생성 흐름이 정상 동작함을 확인했다. 다만 이 모델은 Base 0.5B 모델이므로 역사적 사실성이나 문단의 일관성은 낮고, 예시 결과처럼 반복·비약이 섞일 수 있다. 이 실습의 목적은 답변 품질 비교가 아니라 모델을 한 번 적재한 뒤 API 요청으로 생성하는 서빙 흐름을 확인하는 데 있다.

#### 서버 기동 로그에서 확인한 vLLM 최적화

이번 서버는 단순히 모델만 올린 것이 아니라, 아래 최적화 경로를 준비한 뒤 요청을 처리했다.

| 확인 항목 | 로그에서 확인한 값 | 의미 |
| :--- | :--- | :--- |
| 비동기 스케줄링 | `Asynchronous scheduling is enabled` | 여러 요청을 유휴 슬롯 없이 처리하기 위한 기반이다. |
| Prefix caching·Chunked prefill | 두 기능이 `enabled`; 이후 Prefix cache hit rate 약 32~47% | 같은 접두 프롬프트의 Prefill 결과와 KV Cache를 재사용할 수 있다. |
| Attention·sampling | FlashAttention 2, FlashInfer top-k/top-p sampler | Attention과 샘플링을 GPU 친화적인 구현으로 실행한다. |
| Torch compile | 그래프 컴파일 10.69초 | 반복 실행 경로를 미리 컴파일해 이후 요청의 실행 오버헤드를 줄인다. |
| CUDA Graph | Piecewise 51개, Full 35개 캡처 | 자주 반복되는 GPU 실행 시퀀스를 그래프로 재사용한다. |
| KV Cache 계획 | 사용 가능 7.68GiB, 2,048토큰 기준 이론상 동시 요청 327.8개 | `gpu_memory_utilization=0.80` 안에서 모델 가중치와 활성화 메모리를 제외하고 캐시 공간을 계산한 결과다. |

초기 엔진 준비는 모델 적재·컴파일·그래프 캡처를 포함해 약 77.79초가 걸렸다. 반면 준비가 끝난 뒤의 생성 요청은 0.42초였다. 즉 처음 실행이 오래 걸린 것은 오류가 아니라, 이후 요청을 빠르게 처리하기 위한 콜드 스타트 비용이다.

:::note[수치 해석 시 주의]
`327.8개`는 이 설정에서 KV Cache가 허용하는 이론상 상한이며, 실제로 동시에 327개 요청을 처리했다는 뜻은 아니다. 또한 로그의 평균 생성 처리량 12.8 tokens/s와 Prefix cache hit rate는 일정 구간의 운영 지표다. 정확한 TTFT·TPOT·동시성 성능 비교는 별도의 반복 부하 테스트가 필요하다.
:::

이 로그가 Part 2에서 다룬 PagedAttention·Prefix Cache·Continuous Batching·FlashAttention이 실제 서빙 엔진에서 준비되는 모습을 보여 준다. 개념 설명은 [Part 2: LLM 서빙 실전과 아키텍처 최적화](/space-notes/posts/ai/llm-serving-part-2/)에서 이어서 정리했다.


---

## 6. [5단계] benchmark_serving.py를 활용한 부하 테스트 실습

vLLM의 배치 처리 성능(TPS, TTFT, TPOT)을 측정하기 위해 부하 테스트를 수행합니다.

```bash
# vLLM 레포지토리 클론 (벤치마크 스크립트 획득)
git clone https://github.com/vllm-project/vllm.git
cd vllm/benchmarks

# ShareGPT 데이터셋 다운로드
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# 100개 요청 부하 타격 실습 (초당 5개 요청)
python3 benchmark_serving.py \
  --backend vllm \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset-name sharegpt \
  --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 100 \
  --request-rate 5.0
```

---

## 7. 트러블슈팅 및 인프라 최적화 팁

:::caution[트러블슈팅: 윈도우 WSL2 메모리 및 GPU OOM 조치]
1. **WSL2 메모리 제한 설정 (`.wslconfig`)**:
   - 윈도우 사용자 폴더(`C:\Users\<사용자명>\.wslconfig`)에 파일을 만들고 WSL2가 사용할 호스트 RAM 용량을 설정합니다:
     ```ini
     [wsl2]
     memory=24GB
     processors=8
     ```
   - PowerShell에서 `wsl --shutdown` 후 다시 실행하면 적용됩니다.

3. **`RuntimeError: Error 500: named symbol not found` (WSL2 CUDA 라이브러리 심볼 에러)**:
   - 윈도우 호스트 드라이버 버전(v591+)과 도커 내부 CUDA runtime 간의 심볼 바인딩 꼬임 현상입니다.
   - **조치 1**: 주피터 노트북 1번 셀 상단에 환경 변수를 추가합니다:
     ```python
     import os
     os.environ["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
     ```
   - **조치 2**: 윈도우 PowerShell에서 `wsl --shutdown` 실행 후 Docker 컨테이너를 재시작합니다.
:::


---

## 📚 참고 자료 (References)

2. **NVIDIA Official Docs**, "CUDA on Windows Subsystem for Linux (WSL2) User Guide", [https://docs.nvidia.com/cuda/wsl-user-guide/](https://docs.nvidia.com/cuda/wsl-user-guide/)
3. **NVIDIA Container Toolkit Installation Guide**, [https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
4. **vLLM Official Installation & Docker Guide**, [https://docs.vllm.ai/en/latest/serving/deploying_with_docker.html](https://docs.vllm.ai/en/latest/serving/deploying_with_docker.html)
