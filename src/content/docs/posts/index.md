---
title: "Study Notes Archive"
description: "AI 모델 서빙, 분산 시스템, 인프라 구축 및 엔지니어링 스터디 기록"
---

Space Notes는 인공지능 모델 서빙, 고성능 분산 인프라, 백엔드 및 컴퓨터 사이언스 이론을 직접 실습하고 깊이 있게 정리하는 개인 기술 아카이브입니다.

---

## 1. Hands-on LLM Serving & Optimization 스터디 시리즈

CloudNet Hands-on LLM Serving and Optimization 스터디를 진행하며, 단순 이론을 넘어 실전 프로덕션 관점의 서빙 아키텍처와 최적화 기법을 다룬 연재물입니다.

| 회차 | 아티클 제목 | 핵심 주제 |
| :--- | :--- | :--- |
| **Part 1** | [[Part 1] LLM 서빙의 이해와 구조적 특징](/space-notes/posts/ai/llm-serving-part-1/) | 웹 서버와의 차이, GPU 병렬성, Prefill vs Decode 병목 |
| **Part 2** | [[Part 2] LLM 서빙 실전과 vLLM 최적화](/space-notes/posts/ai/llm-serving-part-2/) | KV Cache 메커니즘, 메모리 단편화, vLLM 핵심 아키텍처 |
| **Part 3** | [[Part 3] 모델 서빙 시스템 설계와 구현](/space-notes/posts/ai/llm-serving-part-3/) | 서빙 파이프라인 아키텍처, Triton/FastAPI 설계 |
| **Part 4** | [[Part 4] 분산 모델 서빙과 RayService](/space-notes/posts/ai/llm-serving-part-4/) | 분산 추론 아키텍처, Ray 클러스터, KubeRay 오케스트레이션 |
| **Part 5** | [[Part 5] LLM 서빙의 하드웨어 기초와 메모리 벽](/space-notes/posts/ai/llm-serving-part-5/) | GPU 4대 사양, FP8/BF16 비트 구조, Memory Wall 병목 계산 |
| **Part 6** | [[Part 6] LLM 핵심 서빙 최적화 기법](/space-notes/posts/ai/llm-serving-part-6/) | Continuous Batching, Chunked Prefill, PagedAttention, Prefix Caching |

---

## 2. 로컬 GPU 환경 구축 & 실습 기록

로컬 머신(Windows 11 + RTX 4070 Laptop 8GB GPU)에서 가상화 환경을 뚫고 LLM 서빙 인프라를 구축한 트러블슈팅 기록입니다.

- [[환경 구축 기록] WSL2·Docker에서 로컬 RTX 4070 GPU로 LLM 실행하기](/space-notes/posts/ai/local-gpu-wsl2-vllm-guide/)
  - WSL2 NVIDIA GPU 드라이버 패스스루, Docker 컨테이너 GPU 마운트 및 vLLM 로컬 서빙
- [[환경 구축 기록] Windows WSL2·kind에서 로컬 RTX 4070 GPU 패스스루와 KubeRay(RayService) 구축하기](/space-notes/posts/ai/local-gpu-wsl2-rayservice-guide/)
  - kind 로컬 쿠버네티스 클러스터에서 NVIDIA GPU Operator, KubeRay Operator 및 RayService 배포

---

## 3. 거대 모델 아키텍처 딥다이브

- [DeepSeek과 Qwen은 거대 모델을 어떻게 가볍게 띄웠을까?](/space-notes/posts/ai/how-chinese-llms-serve-deepseek-qwen/)
  - DeepSeek-V3/R1의 Multi-Head Latent Attention(MLA), MoE(Mixture of Experts)와 Qwen2.5의 GQA 서빙 효율 분석

---

## 4. 사이드바 탐색 팁
- 왼쪽 사이드바의 카테고리 헤더를 클릭하여 접거나 펼칠 수 있습니다.
- 상단 헤더의 `사이드바 접기/열기(☰)` 버튼을 누르면 넓은 화면에서 다이어그램과 표를 집중해서 볼 수 있습니다.
- 키보드 `Ctrl + K` (또는 `Cmd + K`)를 눌러 언제든지 본문 전체를 실시간 검색할 수 있습니다.

