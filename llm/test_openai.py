from openai import OpenAI

if __name__ == '__main__':
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="test123",
    )

    completion = client.chat.completions.create(
        model="Vikhrmodels/Vikhr-Llama3.1-8B-Instruct-R-21-09-24",
        messages=[
            {'role': 'system',
             'content': 'Ты выполняешь задачу планирования шагов для вычисления какого-то математического выражения. '
                        'Дан запрос пользователя, напиши решение задачки по шагам.'},
            {"role": "user", "content": "2 + 2 * 2"}
        ]
    )

    print(completion.choices[0].message.content)
