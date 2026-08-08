# Space Notes 🚀

> **신우주(Space)의 마크다운 기반 기술 스터디 & 개발 문서 아카이브**

오픈소스 [Astro Starlight](https://starlight.astro.build/) 문서 엔진으로 구축된 고성능 마크다운 스터디 블로그입니다.

- **🌐 Live Demo (GitHub Pages):** [https://tlsdnwn55.github.io/space-notes/](https://tlsdnwn55.github.io/space-notes/)
- **📁 로컬 워크스페이스:** `D:\space-notes`

---

## ⚡ 스터디 노트 추가 가이드

`src/content/docs/` 디렉토리에 마크다운(`.md`) 파일만 추가하고 GitHub에 Push 하면 자동으로 GitHub Pages 배포가 시작됩니다.

```bash
# 1. 예시: 프론트엔드 폴더에 새로운 스터디 노트 생성
D:\space-notes\src\content\docs\frontend\my-topic.md

# 2. 변경사항 커밋 & 푸시
git add .
git commit -m "docs: 프론트엔드 스터디 노트 추가"
git push origin main
```

---

## 💻 로컬 개발 서버 실행

```bash
# 디펜던시 설치
npm install

# 로컬 개발 서버 시작 (http://localhost:4321)
npm run dev

# 프로덕션 빌드 테스트
npm run build
```

---

## 🛠️ 주요 기능

- 🌌 **Space Theme Dark Mode**: 심해 및 우주 감성의 커스텀 다크 모드 스타일
- 🔍 **Instant Search**: Pagefind 기반 실시간 마크다운 키워드/태그 검색
- 📚 **자동 목차(TOC) & 사이드바**: 파일 및 폴더 구조에 따른 자동 인덱싱
- 🚀 **GitHub Actions CI/CD**: `main` 브랜치 푸시 시 자동 배포
