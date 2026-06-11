retrieval evaluation은 LLM을 호출하지 않고 검색 결과만 평가한다.
gold file에는 question과 expected_source_contains 값을 적는다.
eval 명령은 각 질문의 retrieved source list를 출력하고 Hit@K와 MRR을 계산한다.
