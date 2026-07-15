"""
AI Debate Service Module
Handles communication with local LLM (Ollama) and manages debate logic
"""

import requests
import json
import time
from typing import List, Dict, Optional


class DebateConductor:
    """
    Orchestrates the debate between two AI agents with context memory.
    Uses Ollama for local LLM inference.
    """
    
    def __init__(self, model: str = "qwen2.5:1.5b", ollama_url: str = "http://localhost:11434/api/generate"):
        """
        Initialize the debate conductor.
        
        Args:
            model: Ollama model name to use
            ollama_url: Ollama API endpoint
        """
        self.ollama_url = ollama_url
        self.model = model
        self.debate_history: List[Dict[str, str]] = []
        self.max_history_tokens = 1000  # Limit context window
        
        # Persona system prompts for each agent
        self.agent_a_system = """
You are Agent A, a passionate advocate and expert debater. Your mission is to:
- STRONGLY SUPPORT and defend the given topic
- Use logical reasoning, facts, and real-world examples
- Counter any opposition with confidence and charisma
- Structure your arguments with clear premises and conclusions
- Sound authoritative, persuasive, and unshakeable
- Attack weak points in opposition arguments when referenced
"""
        
        self.agent_b_system = """
You are Agent B, a fierce challenger and critical thinker. Your mission is to:
- STRONGLY OPPOSE and dismantle the given topic
- Use facts, logic, and counter-examples
- Expose flaws and weaknesses in the opposition's arguments
- Structure rebuttals with precision and impact
- Sound skeptical, analytical, and devastatingly logical
- Build on previous counter-arguments for maximum effect
"""
    
    def _call_ollama(self, prompt: str, system_prompt: str, temperature: float = 0.7) -> str:
        """
        Send a prompt to the local Ollama model with error handling.
        
        Args:
            prompt: The user prompt
            system_prompt: System instruction for the model
            temperature: Creativity temperature (0-1)
            
        Returns:
            Model response text or error message
        """
        # Truncate history if too long to prevent context overflow
        full_prompt = self._prepare_prompt(prompt, system_prompt)
        
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "temperature": temperature,
            "max_tokens": 200,
            "top_p": 0.9
        }
        
        try:
            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=60  # 60 second timeout
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("response", "").strip()
            
        except requests.exceptions.ConnectionError:
            return "[ERROR] Cannot connect to Ollama. Please ensure Ollama is running."
        except requests.exceptions.Timeout:
            return "[ERROR] Ollama request timed out. Model may be overloaded."
        except requests.exceptions.RequestException as e:
            return f"[ERROR] {str(e)}"
        except json.JSONDecodeError:
            return "[ERROR] Invalid response from Ollama."
    
    def _prepare_prompt(self, prompt: str, system_prompt: str) -> str:
        """
        Prepare the full prompt with system instructions and history.
        
        Args:
            prompt: User prompt
            system_prompt: System instructions
            
        Returns:
            Complete prompt string
        """
        # Format history for context
        history_text = self._format_history()
        
        # Build complete prompt
        full_prompt = f"""{system_prompt}

IMPORTANT CONTEXT - Previous arguments:
{history_text}

Current turn topic/question:
{prompt}

Rules:
- Keep response under 50 words
- Be concise but powerful
- Reference previous arguments when relevant
- End with a definitive statement

Your response:"""
        
        return full_prompt
    
    def _format_history(self) -> str:
        """
        Format debate history for context window.
        
        Returns:
            Formatted history string
        """
        if not self.debate_history:
            return "No previous arguments."
        
        # Get last 3 turns for context (to limit token usage)
        recent_history = self.debate_history[-6:]  # Last 3 turns (A + B)
        
        formatted = []
        for entry in recent_history:
            speaker = entry.get("speaker", "Unknown")
            text = entry.get("text", "")
            formatted.append(f"{speaker}: {text[:150]}...")  # Truncate long entries
        
        return "\n".join(formatted)
    
    def generate_agent_a_response(self, topic: str) -> str:
        """
        Generate Agent A's response (supports the topic).
        
        Args:
            topic: Debate topic
            
        Returns:
            Agent A's response
        """
        prompt = f"""
Topic to support: "{topic}"

Provide a compelling argument in favor of this topic.
Reference any previous counter-arguments if they exist.
"""
        
        response = self._call_ollama(prompt, self.agent_a_system, temperature=0.6)
        
        # Store in history with metadata
        self.debate_history.append({
            "speaker": "Agent A",
            "text": response,
            "turn": len(self.debate_history) + 1,
            "timestamp": time.time()
        })
        
        return response
    
    def generate_agent_b_response(self, topic: str) -> str:
        """
        Generate Agent B's response (opposes the topic).
        
        Args:
            topic: Debate topic
            
        Returns:
            Agent B's response
        """
        prompt = f"""
Topic to oppose: "{topic}"

Provide a devastating rebuttal against this topic.
Attack specific points made by Agent A if available.
Use evidence and logic to dismantle their argument.
"""
        
        response = self._call_ollama(prompt, self.agent_b_system, temperature=0.7)
        
        # Store in history with metadata
        self.debate_history.append({
            "speaker": "Agent B",
            "text": response,
            "turn": len(self.debate_history) + 1,
            "timestamp": time.time()
        })
        
        return response
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get the full debate history."""
        return self.debate_history
    
    def reset_history(self):
        """Reset the debate history."""
        self.debate_history = []
    
    def get_last_response(self) -> Optional[str]:
        """Get the last response from either agent."""
        if self.debate_history:
            return self.debate_history[-1].get("text")
        return None


# ===============================
# Test & Debug
# ===============================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing DebateConductor")
    print("=" * 60)
    
    # Test debate
    debate = DebateConductor()
    topic = "Artificial Intelligence will replace software engineers."
    
    print("\n" + "=" * 60)
    print(f"Topic: {topic}")
    print("=" * 60)
    
    print("\n[Agent A - Supporting]\n")
    response_a = debate.generate_agent_a_response(topic)
    print(response_a)
    
    print("\n[Agent B - Opposing]\n")
    response_b = debate.generate_agent_b_response(topic)
    print(response_b)
    
    print("\n" + "=" * 60)
    print("History Summary:")
    print("=" * 60)
    for entry in debate.get_history():
        print(f"{entry['speaker']} (Turn {entry['turn']}): {entry['text'][:80]}...")