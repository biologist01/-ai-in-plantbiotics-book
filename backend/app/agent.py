"""
AI Agent module using Groq API
"""
from groq import Groq
from typing import List, Dict, Any
import re
from app.config import settings
from app.vector_db import VectorDB

class RAGAgent:
    """RAG Agent using Groq for question answering"""
    
    def __init__(self):
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = "llama-3.3-70b-versatile"
        self.vector_db = VectorDB()

    def _expand_book_query(self, query: str) -> List[str]:
        """Expand user queries like 'chapter 3' / 'module 1' into book-friendly search terms."""
        original = (query or "").strip()
        if not original:
            return [""]

        q = original.lower().strip()
        expansions = [original]

        # Map spelled-out numbers (limited set)
        word_to_num = {
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
        }

        # Capture: chapter 3, module 1, chapter three
        m = re.search(r"\b(chapter|module)\s*([0-9]+|one|two|three|four|five|six|seven|eight|nine|ten)\b", q)
        if m:
            kind = m.group(1)
            num_raw = m.group(2)
            num = word_to_num.get(num_raw, num_raw)
            expansions.extend(
                [
                    f"{kind} {num}",
                    f"{kind}-{num}",
                    f"module {num}",
                    f"module-{num}",
                    f"module-{num} intro",
                    f"module {num} overview",
                ]
            )

        # If user says just 'chapter 3' we still want to pull relevant module content.
        if q in {"chapter", "chapters", "module", "modules"}:
            expansions.append("table of contents")

        # Dedupe while keeping order
        seen = set()
        deduped = []
        for e in expansions:
            e_norm = e.strip().lower()
            if e_norm and e_norm not in seen:
                seen.add(e_norm)
                deduped.append(e)
        return deduped or [original]
    
    def retrieve_context(self, query: str, limit: int = 5, min_score: float = 0.3) -> str:
        """Retrieve relevant context from vector database"""
        expanded_queries = self._expand_book_query(query)

        # Search multiple expanded forms and merge results.
        merged: List[Dict[str, Any]] = []
        seen_keys = set()
        per_query_limit = max(2, limit)

        for q in expanded_queries[:6]:
            results = self.vector_db.search(q, limit=per_query_limit)
            for r in results:
                key = (r.get("title", ""), r.get("content", ""))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                merged.append(r)

        # Sort by score descending and keep top N
        merged.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        results = merged[:limit]

        # Filter results by minimum score
        relevant_results = [r for r in results if r.get('score', 0.0) >= min_score]

        if not results:
            return "No context found in the knowledge base."

        # If nothing clears min_score, fall back to top results but label as low relevance.
        use_results = relevant_results if relevant_results else results

        context_parts = []
        if not relevant_results:
            context_parts.append("[Note] Retrieved context has low similarity; answer may be incomplete.\n")

        for i, result in enumerate(use_results, 1):
            context_parts.append(
                f"[Source {i} - {result['title']} (relevance: {result['score']:.2f})]\n{result['content']}\n"
            )

        return "\n".join(context_parts)
    
    def answer_question(self, question: str, selected_text: str = None) -> Dict[str, Any]:
        """Answer a question using RAG"""
        
        # If user has selected specific text, use that as primary context
        if selected_text and len(selected_text.strip()) > 20:
            context = f"User Selected Text:\n{selected_text}\n\n"
            context += f"Additional Context:\n{self.retrieve_context(question, limit=3)}"
        else:
            context = self.retrieve_context(question, limit=5)

        if settings.strict_rag_mode and "No context found in the knowledge base." in context:
            return {
                "answer": "I couldn't find this in the textbook knowledge base yet. Please ask a question about a specific module/chapter, or rephrase with more detail from the book.",
                "context_used": context,
                "model": self.model,
            }
        
        system_prompt = """You are Plant AI Assistant for the book "AI Revolution in Plant Biotechnology".
    Your role is to answer questions ONLY within the scope of this book and its generated documentation (modules/chapters).

Guidelines:
1. Base your answers primarily on the provided context
2. If the context doesn't contain enough information, say so clearly
3. Provide detailed technical explanations when appropriate
4. Use examples from the textbook when relevant
5. Be concise but comprehensive
6. If user selected specific text, prioritize that in your answer"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            
            return {
                "answer": answer,
                "context_used": context,
                "model": self.model
            }
        
        except Exception as e:
            return {
                "answer": f"Error generating response: {str(e)}",
                "context_used": context,
                "model": self.model,
                "error": True
            }
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Multi-turn conversation support"""
        try:
            # Get the latest user message for context retrieval
            latest_user_msg = next(
                (m["content"] for m in reversed(messages) if m["role"] == "user"),
                ""
            )

            context = self.retrieve_context(latest_user_msg, limit=6)

            if settings.strict_rag_mode and "No context found in the knowledge base." in context:
                return "I couldn't find relevant content for that in the textbook knowledge base yet. Please ask about a specific module/chapter or provide more detail from the book."

            system_msg = {
                "role": "system",
                "content": f"""You are Plant AI Assistant for the book "AI Revolution in Plant Biotechnology".

You MUST keep the conversation focused on: AI in plant biotechnology, agriculture/plant genomics, phenotyping, bioinformatics, plant-focused ML/CV, and the modules/chapters in this book.
If the user asks about unrelated topics (e.g., humanoid robotics), politely steer back to the book and offer related plant-biotech AI help.

Knowledge Base Context:
{context}

Guidelines:
1. Answer questions using the provided context from the knowledge base when available
2. If the user asks for "chapter X" or "module X", interpret it as the book module/chapter and summarize what that module covers based on the knowledge base
3. For follow-up questions (like "explain it", "tell me more"), use the conversation history to understand what the user is referring to
4. If the knowledge base context has low relevance but you can answer from conversation history, do so
5. If you cannot answer, acknowledge what you don't know and ask for clarification
6. Be conversational and helpful while maintaining technical accuracy"""
            }

            full_messages = [system_msg] + messages

            response = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                temperature=0.7,
                max_tokens=1000
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Error in chat: {str(e)}"
