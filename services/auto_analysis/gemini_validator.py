"""Gemini API integration for annotation validation."""

import asyncio
import json
from typing import List, Optional, Dict
import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_MODEL, TIMEOUT_SECONDS


class GeminiValidator:
    """Validates annotations using Gemini API."""

    def __init__(self, api_key: str, guideline: str):
        """Initialize Gemini validator with API key and guideline."""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        self.guideline = guideline
        self.semaphore = asyncio.Semaphore(10)  # Max 10 concurrent requests

    async def validate_batch(self, batch: List[Dict], batch_number: int) -> Dict:
        """Validate a batch of records asynchronously."""
        async with self.semaphore:
            return await self._validate_batch_internal(batch, batch_number)

    async def _validate_batch_internal(self, batch: List[Dict], batch_number: int) -> Dict:
        """Internal method to validate batch using Gemini."""
        try:
            prompt = self._build_validation_prompt(batch)

            # Run API call in executor to avoid blocking
            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None, lambda: self.model.generate_content(prompt)
                ),
                timeout=TIMEOUT_SECONDS,
            )

            # Parse response to extract incorrect items
            incorrect_items = self._parse_validation_response(response.text, batch)

            return {
                "batch_number": batch_number,
                "total_items": len(batch),
                "incorrect_count": len(incorrect_items),
                "incorrect_items": incorrect_items,
                "status": "success",
            }

        except asyncio.TimeoutError:
            return {
                "batch_number": batch_number,
                "status": "timeout",
                "error": f"Timeout after {TIMEOUT_SECONDS}s",
            }
        except Exception as e:
            return {
                "batch_number": batch_number,
                "status": "error",
                "error": str(e),
            }

    def _build_validation_prompt(self, batch: List[Dict]) -> str:
        """Build validation prompt for Gemini."""
        batch_text = ""
        for i, record in enumerate(batch, 1):
            batch_text += f"{i}. ID: {record['id']}\n"
            batch_text += f"   Text: {record['text']}\n"
            batch_text += f"   Predicted Label: {record['predicted_label']}\n\n"

        prompt = f"""You are an expert in Gender Bias annotation according to strict guidelines.

ANNOTATION GUIDELINE:
{self.guideline}

TASK:
Review each record below and identify if the predicted label is INCORRECT according to the guidelines.
Valid labels are: neutral, GB-ATTACK, GB-NORMATIVE, GB-SEX, meta_counter

Return ONLY the records with INCORRECT labels. For each incorrect item, provide:
1. ID
2. Original Text
3. Current (Incorrect) Label
4. Correct Label
5. Brief reason why it's incorrect

If all records are correct, respond with: "ALL_CORRECT"

RECORDS TO VALIDATE:
{batch_text}

RESPONSE FORMAT (JSON):
{{
    "incorrect_records": [
        {{
            "id": "...",
            "text": "...",
            "predicted_label": "...",
            "correct_label": "...",
            "reason": "..."
        }}
    ]
}}

Return ONLY the JSON, no additional text."""

        return prompt

    def _parse_validation_response(self, response_text: str, batch: List[Dict]) -> List[Dict]:
        """Parse Gemini response and extract incorrect items."""
        response_text = response_text.strip()

        # Check if all correct
        if "ALL_CORRECT" in response_text:
            return []

        # Try to parse JSON response
        try:
            # Extract JSON from response (sometimes model adds extra text)
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                data = json.loads(json_str)
                return data.get("incorrect_records", [])
        except json.JSONDecodeError:
            pass

        return []
