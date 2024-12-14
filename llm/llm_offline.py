from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

_PROMPT = [
    {'role': 'system',
     'content': 'Ты выполняешь задачу планирования шагов для вычисления какого-то математического выражения. '
                'Дан запрос пользователя, напиши решение задачки по шагам.'},
    {"role": "user", "content": "2 + 2 * 2"}
]

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('Vikhrmodels/Vikhr-Llama3.1-8B-Instruct-R-21-09-24')
    prompt_template = tokenizer.apply_chat_template(_PROMPT, add_generation_prompt=True)
    llm = LLM(
        model='Vikhrmodels/Vikhr-Llama3.1-8B-Instruct-R-21-09-24',
        gpu_memory_utilization=0.8,
        max_model_len=1024
    )
    outputs = llm.generate(prompt_token_ids=prompt_template, sampling_params=SamplingParams(
        temperature=1.0,
        best_of=4,
        n=1,
        max_tokens=512
    ))
    print(outputs)
