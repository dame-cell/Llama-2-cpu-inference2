import os
from dataclasses import dataclass, asdict
from ctransformers import AutoModelForCausalLM, AutoConfig


@dataclass
class GenerationConfig:
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    max_new_tokens: int
    seed: int
    reset: bool
    stream: bool
    threads: int
    stop: list[str]


def format_prompt(system_prompt: str, user_prompt: str):
   



    system_prompt = f"[INST] <<SYS>>n{system_prompt}<</SYS>>"
    user_prompt = f"{user_prompt}[/INST]"

    return f"{system_prompt}{user_prompt}"


def generate(
    llm: AutoModelForCausalLM,
    generation_config: GenerationConfig,
    system_prompt: str,
    user_prompt: str,
):
    """run model inference, will return a Generator if streaming is true"""

    return llm(
        format_prompt(
            system_prompt,
            user_prompt,
        ),
        **asdict(generation_config),
    )


if __name__ == "__main__":
    config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b", context_length=4096 )
    llm = AutoModelForCausalLM.from_pretrained(
        os.path.abspath("models/llama-2-7b-chat.ggmlv3.q2_K.bin"),
        model_type="llama",
        config=config,
    )

    system_prompt = "A conversation between a user and an LLM-based AI assistant named Local Assistant. Local Assistant gives helpful and honest answers."

    generation_config = GenerationConfig(
        temperature=0.2,
        top_k=0,
        top_p=0.9,
        repetition_penalty=1.0,
        max_new_tokens=512,  # adjust as needed
        seed=42,
        reset=False,  # reset history (cache)
        stream=True,  # streaming per word/token
        threads=int(os.cpu_count() / 2),  # adjust for your CPU
        stop=["[/INST]"],
    )

    user_prefix = "[user]: "
    assistant_prefix = f"[assistant]:"

    while True:
        user_prompt = input(user_prefix)
        generator = generate(llm, generation_config, system_prompt, user_prompt.strip())
        print(assistant_prefix, end=" ", flush=True)
        for word in generator:
            print(word, end="", flush=True)
        print("")
