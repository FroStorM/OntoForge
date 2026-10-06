"""Local model profiles for OntoForge Text2KGBench experiments.

Copy this file to local_models.py and fill in your API credentials there.
local_models.py is intentionally ignored by git.
"""

MODEL_PROFILES = {
    "default": {
        "api_key": "",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "temperature": 0.0,
        "timeout": 90,
        "max_retries": 3,
        "verify_ssl": True,
        "use_environment_fallback": False,
        "cache_enabled": True,
        "cache_dir": ".llm_cache_default",
        "request_sleep": 1.0,
        "retry_sleep": 2.0,
    },
    "strong_a": {
        "api_key": "",
        "base_url": "",
        "model": "",
        "temperature": 0.0,
        "timeout": 120,
        "max_retries": 3,
        "verify_ssl": True,
        "use_environment_fallback": False,
        "cache_enabled": True,
        "cache_dir": ".llm_cache_strong_a",
        "request_sleep": 1.0,
        "retry_sleep": 2.0,
    },
    "mid_a": {
        "api_key": "",
        "base_url": "",
        "model": "",
        "temperature": 0.0,
        "timeout": 120,
        "max_retries": 3,
        "verify_ssl": True,
        "use_environment_fallback": False,
        "cache_enabled": True,
        "cache_dir": ".llm_cache_mid_a",
        "request_sleep": 1.0,
        "retry_sleep": 2.0,
    },
}
