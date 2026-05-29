import os
import time
import json
from dotenv import load_dotenv

load_dotenv()

# --- Model config ---
# Change LLM_PROVIDER in .env to switch providers: groq, openai, anthropic
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


def _get_client():
    """Return the right LLM client based on provider"""
    if LLM_PROVIDER == "groq":
        from groq import Groq
        return Groq(api_key=os.getenv("GROQ_API_KEY"))
    elif LLM_PROVIDER == "openai":
        from openai import OpenAI
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    elif LLM_PROVIDER == "anthropic":
        import anthropic
        return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    else:
        raise ValueError(f"Unknown LLM provider: {LLM_PROVIDER}")


def call_llm(
    prompt: str,
    system: str = None,
    temperature: float = 0.2,
    max_tokens: int = 1000,
    json_mode: bool = False,
    call_type: str = "general",
    stream: bool = False
):
    """
    Central LLM caller. All LLM calls in the app go through here.

    Args:
        prompt: user message
        system: optional system prompt
        temperature: 0.0 for deterministic, higher for creative
        max_tokens: max response tokens
        json_mode: if True, forces JSON output and parses response
        call_type: label for usage tracking (query, extract, summarize, classify etc.)
        stream: if True, returns a generator

    Returns:
        str | dict | generator depending on json_mode and stream
    """
    from llm.usage import log_usage

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    # Retry loop
    for attempt in range(MAX_RETRIES):
        try:
            start_time = time.time()
            client = _get_client()

            if LLM_PROVIDER == "anthropic":
                result = _call_anthropic(client, messages, temperature, max_tokens, stream)
            else:
                result = _call_openai_compatible(client, messages, temperature, max_tokens, stream)

            latency = time.time() - start_time

            if stream:
                return result  # return generator directly

            # Log usage
            log_usage(
                call_type=call_type,
                model=LLM_MODEL,
                prompt_len=len(prompt),
                response_len=len(result),
                latency=latency
            )

            # Parse JSON if requested
            if json_mode:
                return _parse_json(result)

            return result

        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in error_str:
                wait = RETRY_DELAY * (2 ** attempt)
                print(f"Rate limited. Waiting {wait}s before retry {attempt+1}/{MAX_RETRIES}")
                time.sleep(wait)
            else:
                print(f"LLM call failed (attempt {attempt+1}): {e}")
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(RETRY_DELAY)

    raise Exception("LLM call failed after max retries")


def call_llm_stream(prompt: str, system: str = None, temperature: float = 0.2, call_type: str = "stream"):
    """Convenience wrapper for streaming calls"""
    return call_llm(prompt, system=system, temperature=temperature, call_type=call_type, stream=True)


def _call_openai_compatible(client, messages, temperature, max_tokens, stream):
    """Works for both Groq and OpenAI (same API shape)"""
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream
    )
    if stream:
        def generator():
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        return generator()

    return response.choices[0].message.content


def _call_anthropic(client, messages, temperature, max_tokens, stream):
    """Anthropic-specific call"""
    system_msg = None
    user_messages = []
    for m in messages:
        if m["role"] == "system":
            system_msg = m["content"]
        else:
            user_messages.append(m)

    kwargs = {
        "model": LLM_MODEL,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": user_messages
    }
    if system_msg:
        kwargs["system"] = system_msg

    if stream:
        def generator():
            with client.messages.stream(**kwargs) as s:
                for text in s.text_stream:
                    yield text
        return generator()

    response = client.messages.create(**kwargs)
    return response.content[0].text


def call_vision_llm(
    image_path: str,
    prompt: str,
    call_type: str = "vision"
) -> str:
    """
    Call vision LLM to describe an image.
    Returns empty string if no vision model configured.
    """
    vision_provider = os.getenv("VISION_PROVIDER", "").strip()
    vision_model = os.getenv("VISION_MODEL", "").strip()

    if not vision_provider or not vision_model:
        print("No vision model configured — skipping description")
        return ""

    try:
        import base64
        from llm.usage import log_usage

        # Read and encode image
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        # Detect image type
        ext = os.path.splitext(image_path)[1].lower()
        mime_map = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".webp": "image/webp",
            ".tiff": "image/tiff", ".gif": "image/gif"
        }
        mime_type = mime_map.get(ext, "image/jpeg")

        start_time = time.time()

        if vision_provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=vision_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:{mime_type};base64,{image_data}"
                        }}
                    ]
                }],
                max_tokens=500
            )
            description = response.choices[0].message.content

        elif vision_provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            response = client.messages.create(
                model=vision_model,
                max_tokens=500,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": image_data
                            }
                        },
                        {"type": "text", "text": prompt}
                    ]
                }]
            )
            description = response.content[0].text

        else:
            print(f"Vision provider '{vision_provider}' not supported")
            return ""

        latency = time.time() - start_time
        log_usage(
            call_type=call_type,
            model=vision_model,
            prompt_len=len(prompt),
            response_len=len(description),
            latency=latency
        )

        return description

    except Exception as e:
        print(f"Vision LLM call failed: {e}")
        return ""
    
    
def _parse_json(text: str) -> dict:
    """Clean and parse JSON from LLM response"""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return {"error": "Could not parse JSON", "raw": text}