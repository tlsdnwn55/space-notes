---
title: "DB 인덱싱 원리 & B-Tree 쿼리 최적화"
description: "MySQL InnoDB 인덱스 구조, B-Tree 알고리즘, 복합 인덱스 순서 및 커버링 쿼리 최적화"
---

# ⚙️ 데이터베이스 인덱싱(Indexing) 핵심 노하우

> [!TIP]
> 데이터베이스 조회의 성능을 100배 이상 높여주는 **인덱스(Index)의 세부 원리와 B-Tree 렌더링 스캔 구조** 정리 노트입니다.

---

## 1. B-Tree 인덱스 구조

MySQL InnoDB의 기본 인덱스 데이터 구조는 **B-Tree (Balanced Tree)**입니다.
모든 리프 노드(Leaf Node)가 동일한 깊이(Depth)를 유지하여 검색 시간 복잡도 `O(log N)`을 보장합니다.

```sql
-- 인덱스 생성 예시
CREATE INDEX idx_user_status_created 
ON users (status, created_at);
```

---

## 2. 복합 인덱스 (Composite Index) 컬럼 순서 규칙

복합 인덱스를 만들 때 **컬럼의 배치 순서**가 인덱스 타는 여부를 결정합니다 (Leftmost Prefix 법칙).

1. **동등 조건 (`=`, `IN`)**으로 검색되는 컬럼을 앞에 배치
2. **범위 조건 (`<`, `>`, `BETWEEN`, `LIKE`)**으로 검색되는 컬럼은 뒤에 배치

```sql
-- ✅ 올바른 인덱스 활용 (status는 동등 조건, created_at은 범위 조건)
SELECT * FROM users 
WHERE status = 'ACTIVE' 
  AND created_at >= '2026-01-01';
```

---

## 3. 커버링 인덱스 (Covering Index)

테이블 데이터에 직접 접근하지 않고, **인덱스 페이지에 존재하는 데이터만으로 쿼리를 완료**하는 방식입니다. Disk I/O를 획기적으로 줄여줍니다.

```sql
-- 커버링 인덱스 적용 (SELECT절의 컬럼들이 모두 인덱스에 포함됨)
SELECT status, created_at 
FROM users 
WHERE status = 'ACTIVE';
```
