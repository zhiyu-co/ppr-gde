import os


def _endpoint(env_name: str, host_env: str, port_env: str, path: str, default_port: str) -> str:
    explicit = os.getenv(env_name)
    if explicit:
        return explicit
    host = os.getenv(host_env, "127.0.0.1")
    port = os.getenv(port_env, default_port)
    return f"http://{host}:{port}{path}"


VLLM_API_BASE = _endpoint("VLLM_API_BASE", "VLLM_HOST", "VLLM_PORT", "/v1", "8355")
CHAR_API_BASE = _endpoint("CHAR_API_BASE", "CHAR_RM_HOST", "CHAR_RM_PORT", "/score", "8000")
EMBEDDING_API_BASE = _endpoint("EMBEDDING_API_BASE", "EMBEDDING_HOST", "EMBEDDING_PORT", "/encode_batch", "8356")
