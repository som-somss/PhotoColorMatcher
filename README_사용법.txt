# Photo Color Matcher — Windows EXE 자동 빌드

이 패키지는 GitHub Actions의 Windows 환경에서 `PhotoColorMatcher.exe`를 자동으로 만듭니다.
내 PC에 Python을 설치할 필요가 없습니다.

## 빌드 순서
1. GitHub에 로그인합니다.
2. 새 Repository를 만듭니다.
3. 이 ZIP의 압축을 풉니다.
4. 숨김 폴더인 `.github`까지 포함하여 모든 파일/폴더를 Repository에 업로드합니다.
5. Repository 상단의 `Actions` 탭을 엽니다.
6. 왼쪽에서 `Build Windows EXE`를 선택합니다.
7. `Run workflow` 버튼을 누르고 다시 `Run workflow`를 누릅니다.
8. 빌드가 끝나고 초록색 체크가 표시되면 해당 실행 기록을 엽니다.
9. 페이지 아래 `Artifacts`에서 `PhotoColorMatcher-Windows`를 받습니다.
10. 받은 ZIP을 풀면 `PhotoColorMatcher.exe`가 있습니다.

## 프로그램 사용
1. PhotoColorMatcher.exe 실행
2. 기준 사진 선택
3. 보정할 사진을 파일 또는 폴더 단위로 추가
4. 출력 폴더 지정
5. 색감 매칭 강도 선택
6. 자동 색감 보정 시작

홈페이지 현장사진은 80~85% 강도를 먼저 권장합니다.
원본은 수정하지 않고 결과 파일 이름에 `_matched`를 붙여 별도 저장합니다.
