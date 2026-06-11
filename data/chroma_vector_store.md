Chroma는 이 프로젝트의 persistent vector store로 사용된다.
ingest 명령은 로컬 문서를 chunk로 나누고 embedding을 만든 뒤 Chroma collection에 저장한다.
retrieve 명령은 저장된 Chroma collection에서 질문과 가까운 문서 조각을 top-k로 가져온다.
