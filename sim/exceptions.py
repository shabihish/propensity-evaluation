class InternalServiceError(RuntimeError):
    def __init__(self, message: str):
        super().__init__(f"Call successful but execution failed: {message}")
