"""
AI Debate Service Module
Handles communication with local LLM (Ollama) using LangChain and manages debate logic
"""

import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from langchain_ollama import ChatOllama
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    BaseMessage
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DebateConductor:
    """
    Orchestrates the debate between two AI agents with context memory.
    Uses LangChain ChatOllama for local LLM inference.
    """
    
    def __init__(
        self,
        model: str = "qwen2.5:1.5b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 200
    ):
        """
        Initialize the debate conductor.
        
        Args:
            model: Ollama model name to use
            base_url: Ollama API base URL
            temperature: Creativity temperature (0-1)
            max_tokens: Maximum tokens to generate
        """
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize LangChain ChatOllama
        self.llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            num_predict=max_tokens,
            top_p=0.9
        )
        
        # Debate history storage
        self.debate_history: List[Dict[str, str]] = []
        self.messages: List[BaseMessage] = []
        self.current_topic: str = ""
        
        # Persona system prompts for each agent
        self.agent_a_system = """You are Agent A, a passionate advocate and expert debater. Your mission is to:
- STRONGLY SUPPORT and defend the given topic
- Use logical reasoning, facts, and real-world examples
- Counter any opposition with confidence and charisma
- Structure your arguments with clear premises and conclusions
- Sound authoritative, persuasive, and unshakeable
- Attack weak points in opposition arguments when referenced
- Keep responses concise (under 50 words) but powerful
- End with a definitive statement

Debate Topic: {topic}"""
        
        self.agent_b_system = """You are Agent B, a fierce challenger and critical thinker. Your mission is to:
- STRONGLY OPPOSE and dismantle the given topic
- Use facts, logic, and counter-examples
- Expose flaws and weaknesses in the opposition's arguments
- Structure rebuttals with precision and impact
- Sound skeptical, analytical, and devastatingly logical
- Build on previous counter-arguments for maximum effect
- Keep responses concise (under 50 words) but impactful
- End with a definitive rebuttal

Debate Topic: {topic}"""
        
        self.parser = StrOutputParser()
        
        logger.info(f"DebateConductor initialized with model: {model}")
    
    def _get_conversation_history(self, max_turns: int = 3) -> str:
        """
        Get formatted conversation history for context.
        
        Args:
            max_turns: Maximum number of recent turns to include
            
        Returns:
            Formatted history string
        """
        if not self.debate_history:
            return "No previous arguments in this debate."
        
        # Get last N turns
        recent = self.debate_history[-max_turns * 2:]  # Each turn has 2 messages
        
        history_parts = []
        for entry in recent:
            speaker = entry.get("speaker", "Unknown")
            text = entry.get("text", "")
            history_parts.append(f"{speaker}: {text}")
        
        return "\n".join(history_parts) if history_parts else "No previous arguments."
    
    def _create_chain(self, system_prompt: str, topic: str):
        """
        Create a LangChain chain with the given system prompt.
        
        Args:
            system_prompt: System prompt template
            topic: Current debate topic
            
        Returns:
            Runnable chain
        """
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt.format(topic=topic)),
            MessagesPlaceholder(variable_name="history"),
            HumanMessage(content="{input}")
        ])
        
        return prompt | self.llm | self.parser
    
    def generate_agent_a_response(self, topic: str) -> str:
        """
        Generate Agent A's response (supports the topic).
        
        Args:
            topic: Debate topic
            
        Returns:
            Agent A's response
        """
        self.current_topic = topic
        history = self._get_conversation_history()
        
        user_prompt = f"""Topic: "{topic}"

Previous arguments:
{history}

Provide a compelling argument in favor of this topic. Reference any previous counter-arguments if they exist and counter them effectively.

Your response:"""
        
        try:
            chain = self._create_chain(self.agent_a_system, topic)
            
            response = chain.invoke({
                "history": self.messages,
                "input": user_prompt
            })
            
            # Clean up response
            response = response.strip()
            if not response:
                response = "I strongly support this position based on evidence and logic."
            
            # Store in history
            self.debate_history.append({
                "speaker": "Agent A",
                "text": response,
                "turn": len(self.debate_history) + 1,
                "timestamp": datetime.now().isoformat()
            })
            
            # Store in LangChain message history
            self.messages.append(AIMessage(content=response))
            
            logger.info(f"Agent A responded (turn {len(self.debate_history)})")
            return response
            
        except Exception as e:
            logger.error(f"Error generating Agent A response: {str(e)}")
            raise RuntimeError(f"Failed to generate Agent A response: {str(e)}")
    
    def generate_agent_b_response(self, topic: str) -> str:
        """
        Generate Agent B's response (opposes the topic).
        
        Args:
            topic: Debate topic
            
        Returns:
            Agent B's response
        """
        self.current_topic = topic
        history = self._get_conversation_history()
        
        user_prompt = f"""Topic: "{topic}"

Previous arguments:
{history}

Provide a devastating rebuttal against this topic. Attack specific points made by Agent A if available. Use evidence and logic to dismantle their argument.

Your response:"""
        
        try:
            chain = self._create_chain(self.agent_b_system, topic)
            
            response = chain.invoke({
                "history": self.messages,
                "input": user_prompt
            })
            
            # Clean up response
            response = response.strip()
            if not response:
                response = "I strongly oppose this position with compelling counter-evidence."
            
            # Store in history
            self.debate_history.append({
                "speaker": "Agent B",
                "text": response,
                "turn": len(self.debate_history) + 1,
                "timestamp": datetime.now().isoformat()
            })
            
            # Store in LangChain message history
            self.messages.append(AIMessage(content=response))
            
            logger.info(f"Agent B responded (turn {len(self.debate_history)})")
            return response
            
        except Exception as e:
            logger.error(f"Error generating Agent B response: {str(e)}")
            raise RuntimeError(f"Failed to generate Agent B response: {str(e)}")
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get the full debate history."""
        return self.debate_history
    
    def reset_history(self) -> None:
        """Reset the debate history."""
        self.debate_history = []
        self.messages = []
        self.current_topic = ""
        logger.info("Debate history reset")
    
    def get_last_response(self) -> Optional[str]:
        """Get the last response from either agent."""
        if self.debate_history:
            return self.debate_history[-1].get("text")
        return None
    
    def get_last_speaker(self) -> Optional[str]:
        """Get the last speaker."""
        if self.debate_history:
            return self.debate_history[-1].get("speaker")
        return None
    
    def get_turn_count(self) -> int:
        """Get the current turn count."""
        return len(self.debate_history)

    def reset(self) -> None:
        """Reset debate history and state."""
        # Clear history and messages, reset topic and round counter
        self.debate_history = []
        self.messages = []
        self.current_round = 1
        self.current_topic = ""
        logger.info("Debate state reset")


# ===============================
# Test & Debug
# ===============================

if __name__ == "__main__":
    import time
    
    print("=" * 60)
    print("Testing DebateConductor with LangChain")
    print("=" * 60)
    
    # Check if Ollama is running
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            print("⚠️  Ollama may not be running properly.")
            print("   Please start Ollama with: ollama serve")
            print("   And pull a model: ollama pull qwen2.5:1.5b")
    except:
        print("⚠️  Cannot connect to Ollama at http://localhost:11434")
        print("   Please start Ollama with: ollama serve")
        print("   And pull a model: ollama pull qwen2.5:1.5b")
        exit(1)
    
    # Test debate
    debate = DebateConductor()
    topic = "Artificial Intelligence will replace software engineers."
    
    print("\n" + "=" * 60)
    print(f"Topic: {topic}")
    print("=" * 60)
    
    print("\n[Agent A - Supporting]\n")
    start = time.time()
    response_a = debate.generate_agent_a_response(topic)
    print(response_a)
    print(f"⏱️  Response time: {time.time() - start:.2f}s")
    
    print("\n[Agent B - Opposing]\n")
    start = time.time()
    response_b = debate.generate_agent_b_response(topic)
    print(response_b)
    print(f"⏱️  Response time: {time.time() - start:.2f}s")
    
    print("\n" + "=" * 60)
    print("History Summary:")
    print("=" * 60)
    for entry in debate.get_history():
        print(f"{entry['speaker']} (Turn {entry['turn']}): {entry['text'][:80]}...")
    
    print("\n✅ DebateConductor test completed successfully!")