import os
from typing import List, Dict, Any
from langchain_google_vertexai import ChatVertexAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor, create_tool_calling_agent # Corrected import
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI
from googletrans import Translator

class AwazSevaAgent:
    def __init__(self):
        # Initialize components
        self.translator = Translator()
        self.llm = self._setup_llm()
        self.knowledge_base = self._setup_knowledge_base()
        self.web_search_tool = self._setup_web_search()
        self.agent = self._create_agent()
        
    def _setup_llm(self):
        """Setup Google Gemini LLM."""
        return GoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7
        )
    
    def _setup_knowledge_base(self):
        """Setup FAISS vector store with local documents."""
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Check if a pre-built index exists to avoid re-processing
        if os.path.exists("faiss_index"):
            return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) # Recommended for local dev

        documents = self._load_local_documents()
        
        if not documents:
            documents = [
                Document(page_content="This is a sample knowledge base entry about government services.", 
                         metadata={"source": "sample"}),
            ]
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        vectorstore = FAISS.from_documents(splits, embeddings)
        vectorstore.save_local("faiss_index") # Save the index for future use
        return vectorstore
    
    def _load_local_documents(self):
        """Load local documents from docs/ directory."""
        docs = []
        docs_dir = "docs/"
        
        if os.path.exists(docs_dir):
            for filename in os.listdir(docs_dir):
                if filename.endswith('.txt'):
                    with open(os.path.join(docs_dir, filename), 'r', encoding='utf-8') as f:
                        content = f.read()
                        docs.append(Document(
                            page_content=content,
                            metadata={"source": filename}
                        ))
        return docs
    
    def _setup_web_search(self):
        """Setup Tavily web search tool."""
        return TavilySearchResults(
            api_key=os.getenv("TAVILY_API_KEY"),
            max_results=3,
            search_depth="basic"
        )
    
    def _create_knowledge_base_tool(self):
        """Create knowledge base search tool."""
        def search_knowledge_base(query: str) -> str:
            """Search the local knowledge base for relevant information."""
            retriever = self.knowledge_base.as_retriever(search_kwargs={"k": 3})
            docs = retriever.get_relevant_documents(query)
            
            if docs:
                results = [f"Content: {doc.page_content}\nSource: {doc.metadata.get('source', 'unknown')}" for doc in docs]
                return "\n\n".join(results)
            return "No relevant information found in knowledge base."
        
        return Tool(
            name="knowledge_base_search",
            description="Search the local knowledge base for information about government services, citizen services, and documentation. Use this for questions about established procedures, policies, and services.",
            func=search_knowledge_base
        )
    
    def _create_agent(self):
        """Create LangChain agent with tools."""
        kb_tool = self._create_knowledge_base_tool()
        
        tools = [
            kb_tool,
            Tool(
                name="web_search",
                description="Search the web for current information, news, or real-time updates. Use this for recent events, current status, or information not in the knowledge base.",
                func=self.web_search_tool.run
            )
        ]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are Awaaz Seva, a helpful assistant for government and citizen services in India.
            
            Your job is to help users with their queries by:
            1. First checking if the information is available in the knowledge base (for established procedures, policies)
            2. If not found or if current information is needed, search the web
            3. Provide clear, helpful responses in the same language as the user's query

            Always be polite and helpful. If you cannot find specific information, acknowledge it and suggest general guidance."""),
            MessagesPlaceholder(variable_name="chat_history"), # Added for conversation memory
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Use the correct agent type for Google's models
        agent = create_tool_calling_agent(self.llm, tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5 # Increased for better agent behavior
        )
    
    def detect_language(self, text: str) -> str:
        """Detect if text is in Hindi."""
        try:
            return self.translator.detect(text).lang
        except:
            return 'en'
    
    def translate_text(self, text: str, target_lang: str) -> str:
        """Translate text to target language."""
        try:
            return self.translator.translate(text, dest=target_lang).text
        except:
            return text
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process user query with language handling."""
        detected_lang = self.detect_language(query)
        
        if detected_lang == 'hi':
            processed_query = self.translate_text(query, 'en')
        else:
            processed_query = query
        
        try:
            # Pass the input to the agent
            response = self.agent.invoke({"input": processed_query, "chat_history": []})
            answer = response.get("output", "I couldn't process your query.")
        except Exception as e:
            answer = f"I encountered an error while processing your query: {str(e)}"
        
        # Translate the final answer back if the original query was Hindi
        if detected_lang == 'hi':
            answer = self.translate_text(answer, 'hi')
        
        return {
            "original_query": query,
            "detected_language": detected_lang,
            "processed_query": processed_query,
            "answer": answer
        }

# Global agent instance
_agent_instance = None

def get_agent():
    """Get or create agent instance."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = AwazSevaAgent()
    return _agent_instance