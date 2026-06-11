LangChain은 이 프로젝트에서 RAG orchestration을 담당한다.
retrieval 단계에서 가져온 문서 조각을 prompt context로 구성하고, ChatOpenAI-compatible client를 통해 Lemonade Server에 전달한다.
이 프로젝트는 LangGraph나 agentic workflow 없이 단순한 LangChain 흐름만 검증한다.
