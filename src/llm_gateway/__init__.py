import os


def main() -> None:
    import uvicorn

    from llm_gateway import logs

    logs.setup(os.environ.get("LOG_LEVEL", "INFO"))

    uvicorn.run(
        "llm_gateway.app:create_app",
        factory=True,
        host=os.environ.get("GATEWAY_HOST", "127.0.0.1"),
        port=int(os.environ.get("GATEWAY_PORT", "8000")),
        log_config=None,
    )
