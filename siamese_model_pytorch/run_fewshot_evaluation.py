import argparse
import asyncio
import json
import logging
import os
from typing import Any, Dict

import aiohttp
import pandas as pd
from tqdm.asyncio import tqdm


# --- Utility Functions ---

def load_cache(cache_file: str) -> Dict[str, Any]:
    """Load cache from a JSON file."""
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(f"Could not load cache file {cache_file}: {e}")
    return {}


def save_cache(cache: Dict[str, Any], cache_file: str):
    """Save cache to a JSON file."""
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except IOError as e:
        logging.error(f"Could not save cache file {cache_file}: {e}")


def get_record_details_for_prompt(row: pd.Series, suffix: str) -> str:
    """Extract record details from a DataFrame row for the prompt."""
    details = []
    prefix_map = {'1': ['bib1_', 'music1_', 'person1_'], '2': ['bib2_', 'music2_', 'person2_']}
    prefixes = prefix_map.get(suffix, [])

    for col, val in row.items():
        for prefix in prefixes:
            if col.startswith(prefix) and pd.notna(val) and val != '':
                # 'bib1_title' -> 'title'
                key = col[len(prefix):]
                details.append(f"- {key}: {val}")
                break # Move to next column once prefix is matched

    if not details:
        return "No information available."
    return "\n".join(details)


# --- Core Evaluation Logic ---

async def get_llm_judgment(
    session: aiohttp.ClientSession,
    record1_details: str,
    record2_details: str,
    system_prompt: str,
    fewshot_examples: list,
    cache: Dict[str, Any],
    model: str,
    pair_id: str,
    retries: int = 3,
    delay: int = 5
) -> Dict[str, Any]:
    """Get LLM judgment for a pair of records, with caching and retries."""
    cache_key = f"{pair_id}_{model}"
    if cache_key in cache:
        return cache[cache_key]

    messages = [
        {"role": "system", "content": system_prompt},
        *fewshot_examples,
        {"role": "user", "content": f"Record 1:\n{record1_details}\n\nRecord 2:\n{record2_details}"}
    ]

    payload = {"model": model, "messages": messages, "temperature": 0}
    headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}

    for attempt in range(retries):
        try:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            ) as response:
                response.raise_for_status()
                result = await response.json()
                response_text = result["choices"][0]["message"]["content"].strip()

                is_similar = "はい" in response_text.split('\n')[0].strip().lower()
                score = 1.0 if is_similar else 0.0 # Default score

                try:
                    score_line = next(
                        line for line in response_text.split('\n')
                        if "類似度スコア:" in line
                    )
                    score = float(score_line.split("類似度スコア:")[1].strip())
                except (StopIteration, ValueError, IndexError):
                    logging.warning(
                        f"Could not parse score for {pair_id}. "
                        f"Defaulting to {'1.0' if is_similar else '0.0'}."
                    )

                parsed_result = {
                    "is_similar": is_similar,
                    "score": score,
                    "raw_response": response_text
                }
                cache[cache_key] = parsed_result
                return parsed_result

        except aiohttp.ClientError as e:
            logging.warning(
                f"API call for {pair_id} failed (attempt {attempt + 1}/{retries}): {e}"
            )
            if attempt < retries - 1:
                await asyncio.sleep(delay * (attempt + 1))
            else:
                error_msg = f"API Error after {retries} attempts: {e}"
                return {"is_similar": None, "score": None, "error": error_msg}
        except asyncio.TimeoutError:
            logging.warning(f"Timeout for pair {pair_id} (attempt {attempt + 1}/{retries}).")
            if attempt < retries - 1:
                 await asyncio.sleep(delay)
            else:
                error_msg = "Exhausted retries due to timeout."
                return {"is_similar": None, "score": None, "error": error_msg}

    return {"is_similar": None, "score": None, "error": "Exhausted all retries."}


async def process_all_pairs(
    df: pd.DataFrame,
    system_prompt: str,
    fewshot_examples: list,
    cache: Dict[str, Any],
    max_workers: int,
    model: str
):
    """Process all record pairs asynchronously."""
    tasks = []
    connector = aiohttp.TCPConnector(limit=max_workers)
    async with aiohttp.ClientSession(connector=connector) as session:
        for _, row in df.iterrows():
            pair_id = f"{row['record_id_1']}_{row['record_id_2']}"
            record1_details = get_record_details_for_prompt(row, "1")
            record2_details = get_record_details_for_prompt(row, "2")
            
            task = get_llm_judgment(
                session, record1_details, record2_details, system_prompt,
                fewshot_examples, cache, model, pair_id
            )
            tasks.append(task)
        
        return await tqdm.gather(*tasks, desc="Evaluating pairs")


async def main():
    """Main function to run the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate LLM judgment on record pairs using few-shot examples."
    )
    parser.add_argument(
        "--input_file", type=str, required=True,
        help="Path to the input CSV file with record pairs."
    )
    parser.add_argument(
        "--fewshot_examples_file", type=str, required=True,
        help="Path to the JSON file with few-shot examples."
    )
    parser.add_argument(
        "--output_file", type=str, required=True,
        help="Path to the output CSV file."
    )
    parser.add_argument(
        "--cache_file", type=str, default="llm_api_cache.json",
        help="Path to the cache file for LLM API calls."
    )
    parser.add_argument(
        "--max_workers", type=int, default=10,
        help="Number of concurrent workers for API calls."
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini",
        help="The model to use for evaluation."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
    )

    if not os.getenv("OPENAI_API_KEY"):
        logging.error("OPENAI_API_KEY environment variable not set.")
        return

    cache = load_cache(args.cache_file)
    
    try:
        with open(args.fewshot_examples_file, 'r', encoding='utf-8') as f:
            fewshot_data = json.load(f)
    except FileNotFoundError:
        logging.error(f"Few-shot file not found: {args.fewshot_examples_file}")
        return
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {args.fewshot_examples_file}: {e}")
        return

    system_prompt = fewshot_data.get("system_prompt")
    fewshot_examples = fewshot_data.get("examples", [])

    if not system_prompt:
        logging.error("'system_prompt' not found in the few-shot examples file.")
        return

    try:
        pairs_df = pd.read_csv(args.input_file)
    except FileNotFoundError:
        logging.error(f"Input file not found: {args.input_file}")
        return

    results = await process_all_pairs(
        pairs_df, system_prompt, fewshot_examples,
        cache, args.max_workers, args.model
    )
    
    results_df = pd.DataFrame(results)
    output_df = pd.concat([pairs_df.reset_index(drop=True), results_df], axis=1)
    
    output_df.to_csv(args.output_file, index=False)
    save_cache(cache, args.cache_file)
    logging.info(f"Evaluation complete. Results saved to {args.output_file}")


if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())