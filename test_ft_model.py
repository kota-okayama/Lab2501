import os
import asyncio
from openai import AsyncOpenAI

async def test_ft_model():
    """
    指定されたファインチューニングモデルのAPI呼び出しをテストする。
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("エラー: 環境変数 OPENAI_API_KEY が設定されていません。")
        return

    # ファインチューニング後のモデルID (正しいものに修正)
    ft_model_id = "ft:gpt-4o-mini-2024-07-18:mlab:bib-matching-inconsistency-0519:BYiGHy7V"
    
    print(f"テスト開始: モデル '{ft_model_id}'")
    print(f"使用するAPIキーの末尾: ...{api_key[-4:]}")

    client = AsyncOpenAI(api_key=api_key)

    try:
        print("APIを呼び出しています...")
        completion = await client.chat.completions.create(
            model=ft_model_id,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, this is a test."}
            ],
            temperature=0.0,
            max_tokens=20
        )
        print("\n--- 成功 ---")
        print("API呼び出しに成功しました！")
        print("応答:", completion.choices[0].message.content)

    except Exception as e:
        print("\n--- エラー発生 ---")
        print("API呼び出し中にエラーが発生しました。")
        print(f"エラーの型: {type(e)}")
        print(f"エラー詳細:\n{e}")

if __name__ == "__main__":
    asyncio.run(test_ft_model()) 