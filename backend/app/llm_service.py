from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from typing import List, Dict
import os

class LLMService:
    """Handle LLM inference for RAG"""
    
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        
        print(f" Loading LLM model: {model_name}")
        print(" This may take a few minutes on first download...")
        
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Get HF token if available
        hf_token = os.getenv("HF_TOKEN")
        
        try:
            # Load tokenizer with specific settings for Mistral
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                token=hf_token,
                use_fast=True  # Explicitly use fast tokenizer
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("Tokenizer loaded successfully")
            
        except Exception as e:
            print(f" Tokenizer error: {e}")
            print("Trying with use_fast=False...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                token=hf_token,
                use_fast=False  # Fallback to slow tokenizer
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with optimization for CPU/limited resources
        print("Loading model weights (this will take several minutes)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            token=hf_token
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        print(f"LLM model loaded on {self.device}")
    
    def generate_response(
        self, 
        query: str, 
        context: str, 
        max_length: int = 512,
        temperature: float = 0.7
    ) -> str:
        
        # Create prompt with context
        prompt = self._create_prompt(query, context)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=2048,
            add_special_tokens=True
        )
        inputs = inputs.to(self.device)
        
        # Generate with appropriate settings
        with torch.no_grad():
            # Adjust parameters based on model
            if "TinyLlama" in self.model_name or "tinyllama" in self.model_name.lower():
                # TinyLlama works better with these settings
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=min(max_length, 256),  # TinyLlama works better with shorter outputs
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.95,
                    top_k=40,
                    repetition_penalty=1.3,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            else:
                # Mistral/larger models
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
        
        # Decode - only the generated part, not the input
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the answer part (remove the prompt)
        answer = self._extract_answer(full_response, prompt)
        
        return answer
    
    def _create_prompt(self, query: str, context: str) -> str:
        
        # Check which model is being used and format accordingly
        if "TinyLlama" in self.model_name or "tinyllama" in self.model_name.lower():
            # TinyLlama format - simpler is better
            prompt = f"""<|system|>
You are a helpful assistant. Answer the question based on the context provided. Keep your answer concise and accurate.</s>
<|user|>
Context: {context}

Question: {query}</s>
<|assistant|>
"""
        else:
            # Mistral format
            prompt = f"""[INST] You are a helpful AI assistant. Use the following context to answer the user's question accurately. If you cannot find the answer in the context, say so.

Context:
{context}

Question: {query}

Answer: [/INST]"""
        
        return prompt
    
    def _extract_answer(self, full_response: str, prompt: str) -> str:
        
        answer = full_response
        
        # Try to extract answer based on model type
        if "TinyLlama" in self.model_name or "tinyllama" in self.model_name.lower():
            # TinyLlama: Look for content after <|assistant|>
            if "<|assistant|>" in answer:
                answer = answer.split("<|assistant|>")[-1]
            
            # Remove special tokens
            answer = answer.replace("</s>", "")
            answer = answer.replace("<|system|>", "")
            answer = answer.replace("<|user|>", "")
            answer = answer.replace("<|assistant|>", "")
            
        else:
            # Mistral: Look for content after [/INST]
            if "[/INST]" in answer:
                answer = answer.split("[/INST]")[-1]
        
        # Remove any remaining prompt text
        if "Context:" in answer and "Question:" in answer:
            # If the answer still contains the prompt, extract only what comes after
            lines = answer.split("\n")
            # Find where the actual answer starts (after Question:)
            answer_lines = []
            found_question = False
            for line in lines:
                if "Question:" in line:
                    found_question = True
                    continue
                if found_question and line.strip():
                    answer_lines.append(line)
            if answer_lines:
                answer = "\n".join(answer_lines)
        
        # Final cleanup
        answer = answer.strip()
        
        # If we still have a messy answer, just take the first clean sentence/paragraph
        if len(answer) > 500 or "Context:" in answer:
            sentences = answer.split(".")
            clean_sentences = [s.strip() for s in sentences if s.strip() and "Context" not in s and "Question" not in s]
            if clean_sentences:
                answer = ". ".join(clean_sentences[:3]) + "."
        
        return answer if answer else "I couldn't generate a proper answer based on the context provided."
    
