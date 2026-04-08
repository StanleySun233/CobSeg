class LlmRunStats:
    def __init__(self):
        self.total_requests = 0
        self.success = 0
        self.fail = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def record(self, result):
        if not isinstance(result, dict):
            return
        self.total_requests += int(result.get("attempts", 0))
        self.input_tokens += int(result.get("input_tokens", 0))
        self.output_tokens += int(result.get("output_tokens", 0))
        if result.get("success", False):
            self.success += 1
        else:
            self.fail += 1

    def total_tokens(self):
        return self.input_tokens + self.output_tokens

    def desc_line(self):
        total_tok = self.total_tokens()
        return (
            f"tokens in={self.input_tokens} out={self.output_tokens} total={total_tok} | "
            f"requests={self.total_requests} | ok={self.success} | fail={self.fail}"
        )
