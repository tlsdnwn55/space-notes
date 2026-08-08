---
title: "React 19 & 웹 성능 최적화 노트"
description: "React Virtual DOM 렌더링 원리, React 19 Compiler, Server Components 및 memo/useCallback 최적화 전략"
---

# 🎨 React 렌더링 최적화 & 최신 트렌드

> [!IMPORTANT]
> React 애플리케이션의 불필요한 리렌더링(Re-render)을 방지하고 최신 **React 19**의 변경 사항 및 서버 컴포넌트(RSC) 개념을 정리한 노트입니다.

---

## 1. React 렌더링 2단계 (Render Phase vs Commit Phase)

React에서 UI가 업데이트되는 과정은 크게 두 단계로 나누어집니다.

1. **Render Phase**: 컴포넌트를 호출하여 Virtual DOM 트리를 생성하고 이전 트리와의 차이점(Diffing)을 계산하는 단계 (Pure Function이어야 함)
2. **Commit Phase**: 계산된 차이점을 실제 DOM에 반영하는 단계

```typescript
// useMemo 및 useCallback을 통한 불필요한 연산/함수 재생성 방지
import React, { useState, useMemo, useCallback } from 'react';

interface ComponentProps {
  items: number[];
  onSelect: (item: number) => void;
}

export const FilteredList: React.FC<ComponentProps> = ({ items, onSelect }) => {
  const [filter, setFilter] = useState('');

  // 1. 값 캐싱: items가 변경될 때만 재계산
  const filteredItems = useMemo(() => {
    return items.filter(item => item.toString().includes(filter));
  }, [items, filter]);

  // 2. 함수 참조 유지: 자식 컴포넌트 리렌더링 방지
  const handleSelect = useCallback((item: number) => {
    onSelect(item);
  }, [onSelect]);

  return (
    <div>
      <input value={filter} onChange={(e) => setFilter(e.target.value)} />
      <ul>
        {filteredItems.map(item => (
          <li key={item} onClick={() => handleSelect(item)}>{item}</li>
        ))}
      </ul>
    </div>
  );
};
```

---

## 2. React 19 Forget Compiler (자동 메모이제이션)

React 19에서는 더 이상 수동으로 `useMemo`, `useCallback`, `React.memo`를 작성하지 않아도 **React Compiler**가 빌드 타임에 자동으로 종속성을 추적하여 리렌더링을 최적화해 줍니다.

```jsx
// React 19 컴파일러 도입 후: 수동 useMemo 없이도 자동 메모이제이션 처리됨
function FriendList({ friends }) {
  const onlineFriends = friends.filter(friend => friend.isOnline);
  return <List items={onlineFriends} />;
}
```

---

## 3. 웹 성능 최적화 체크리스트

- [x] LCP(Largest Contentful Paint) 개선: 중요한 이미지 `priority` 및 `preload` 적용
- [x] CLS(Cumulative Layout Shift) 방지: 이미지 및 스켈레톤 UI 높이 값 명시
- [x] Code Splitting: `React.lazy()` 및 `Suspense` 적용
