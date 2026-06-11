Lemonade Server는 로컬 LLM을 OpenAI-compatible API 형태로 제공한다.
이 프로젝트의 check 명령은 Lemonade Server의 /models 엔드포인트를 호출해 서버 접근성과 설정된 chat model 존재 여부를 확인한다.
ask 명령은 검색된 context를 Lemonade Server에 보내 source-grounded answer를 생성한다.
