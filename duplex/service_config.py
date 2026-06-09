
# ============== Deprecated backend aliases ================
# The Sanic gateway now talks to LMDeploy through DUPLEX_GRPC_TARGET. These
# aliases are retained for older imports and should not be used by new code.
BACKEND_HOST = "localhost"
BACKEND_URL = f"{BACKEND_HOST}:8001"
BACKEND_MODEL_NAME = "tensorrt_llm_bls"

IP = BACKEND_HOST
URL = BACKEND_URL
MODEL_NAME = BACKEND_MODEL_NAME

# ============== LLM ================
system_prompt = "<|im_start|>system\nYou are a helpful assistant."
assistant_prompt = "<|im_end|>\n<|im_start|>assistant\n"
user_prompt = "<|im_end|>\n<|im_start|>user\n{}"
trunc_delimiter = "<|im_end|>\n<|im_start|>assistant\n"

END_PROB_THRESHOLD = 1e-1
CERTAIN_PROB_THRESHOLD = 5e-2
INVALID_BIAS = 1e-1

warmup_prompt = [
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>",
] * 5

# ============== Basic ================
INVALID_START = "["
INVALID_END = "]"
ALIGN_TOKEN = "-"
SPECIAL_TEXT = "CEM_NULL"
MIN_INFER_NUM_WORD = 2
MAX_BACK = 5
PARTICLES = ["了", "啊", "呀", "呢", "吧", "吗", "嘛", "啦", "的"]
BACKCHANNEL_WORDS = ["嗯", "啊", "呃", "诶", "喂", "好", "对", "好的", "对的", "你好", "您好"]
