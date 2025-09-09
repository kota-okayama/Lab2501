import openai
import os

def test_model(model_id, client):
    print(f"\n--- Testing model: {model_id} ---")
    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, who are you?"}
            ],
            temperature=0.0,
            max_tokens=50
        )
        response_text = completion.choices[0].message.content.strip()
        print(f"  ✅ Success!")
        print(f"  Response: {response_text}")
    except Exception as e:
        print(f"  ❌ Error!")
        print(f"  Details: {e}")

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        return

    client = openai.OpenAI(api_key=api_key)

    base_model = "gpt-4o-mini-2024-07-18"
    ft_model = "ft:gpt-4o-mini-2024-07-18:mlab:wdc-product-inconsistency-100:CDVqqHbd"

    test_model(base_model, client)
    test_model(ft_model, client)

if __name__ == "__main__":
    main()
