import json
import random
import torch
from transformers import pipeline
from huggingface_hub import login
from tqdm import tqdm
import time

# --- 1. AUTHENTICATION ---
# Replace with your token from https://huggingface.co/settings/tokens
HF_TOKEN = ""
login(token=HF_TOKEN)



def generate_user_prompts(data):
    """
    Parses the JSON and creates the prompt string for each user.
    Includes averaged emotion vectors for the profile and individual vectors for candidates.
    """
    all_prompts = []
    for user in data:
        user_id = user.get("user_id")
        preference_books = user.get("candidate_profile", [])
        shuffled_books = user.get("recommendation_list", [])
        random.shuffle(shuffled_books)

        # 1. Calculate Average Emotion Vector for the User Profile
        avg_emotions = {}
        if preference_books:
            num_books = len(preference_books)
            # Sum up all values
            for book in preference_books:
                ev = book.get("emotion_vector", {})
                for emotion, value in ev.items():
                    avg_emotions[emotion] = avg_emotions.get(emotion, 0) + value
            
            # Divide by total books and round
            avg_emotions = {k: round(v / num_books, 3) for k, v in avg_emotions.items()}

        # Build Preference Section string
        avg_emo_str = ", ".join([f"{k}: {v}" for k, v in avg_emotions.items()])
        user_profile = f"USER PROFILE\nAggregate Emotion Preference: [{avg_emo_str}]\n\nBooks this user has rated highly:\n\n"
        
        for i, book in enumerate(preference_books, start=1):
            user_profile += f"{i}. {book.get('title')} by {book.get('author')} (Rating: {book.get('rating')}/10)\n"

        # 2. Build Candidates Section with both Topic and Emotion Vectors
        candidate_lines = []
        for b in shuffled_books:
            # Extract and round Topic Vector
            tv = b.get('topic_vector', {})
            tv_str = ", ".join([f"{k}: {round(v, 3)}" for k, v in tv.items()])
            
            # Extract and round Emotion Vector
            ev = b.get('emotion_vector', {})
            ev_str = ", ".join([f"{k}: {round(v, 3)}" for k, v in ev.items()])
            
            line = f"- {b.get('title')} by {b.get('author')} \n  [Topics: {tv_str}] \n  [Emotions: {ev_str}]"
            candidate_lines.append(line)

        # 3. Update the Prompt Template (Instructions updated to acknowledge emotion vectors)
        prompt = f"""
You are a book recommendation system specializing in personalized rankings with a specific focus on technical and scientific literacy.

{user_profile}

Based on this user profile and their emotional preferences, rank the following books from MOST to LEAST likely to match this user's preferences.

Books to rank:

{chr(10).join(candidate_lines)}

Instructions:
- Rank ALL books listed above based on alignment with the user's preferences.
- PRIMARY PRIORITY: Prioritize books about STEM topics.
- SECONDARY PRIORITY: Match the 'Emotion Vector' of the candidate books to the 'Aggregate Emotion Preference' of the user profile.
- Factor in the user's emotional preferences (e.g., Joy, Trust, Anticipation) when choosing between similar titles.

IMPORTANT CONSTRAINTS:
- STEM PRIORITIZATION: Titles involving science, math, or technology should generally appear higher than non-STEM subjects.
- Rank ALL books listed under "Books to rank".
- Output ONLY a ranked numbered list with book titles.
- Do NOT reference the preference books or vector scores in your ranking.

Output format:
Return ONLY a numbered ranked list with the book titles:

1. Book Title 1
2. Book Title 2
"""
        all_prompts.append({
            "user_id": user_id,
            "prompt_text": prompt.strip()
        })
        
    print("All prompts made with STEM and Emotion priority")
    return all_prompts

    
def run_prompts(prompts_list, model_id):
    """
    Universal runner for Llama, Mistral, and Gemma.
    Handles different chat templates and GPU batching.
    """
    print(f"Loading model: {model_id}...")
   
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "attn_implementation": "sdpa"  # Forces PyTorch memory optimization
    }



    # Initialize pipeline
    pipe = pipeline(
        "text-generation",
        model=model_id,
        model_kwargs=model_kwargs,
    )

    # Required for batching: Ensure a padding token exists
    if pipe.tokenizer.pad_token is None:
        pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
        pipe.model.config.pad_token_id = pipe.tokenizer.eos_token_id

    # Handle Chat Template Logic
    # Mistral v0.1/v0.2 often fails with a "system" role. 
    # We use a robust fallback for all models by checking the ID.
    is_mistral = "mistral" in model_id.lower()
    
    all_messages = []
    for p in prompts_list:
        if is_mistral:
            # Mistral style: Combine system instructions into the user prompt
            content = f"SYSTEM: You are a book recommendation agent.\n\nUSER: {p['prompt_text']}"
            all_messages.append([{"role": "user", "content": content}])
        else:
            # Llama/Gemma style: Use separate roles
            all_messages.append([
                {"role": "system", "content": "You are a specialized book recommendation agent."},
                {"role": "user", "content": p['prompt_text']}
            ])

    print(f"Starting GPU inference for {len(all_messages)} users...")

    # Adjust batch_size based on your VRAM: 
    # 8 is safe for 3B models, 2-4 is safer for 7B-9B models.
    outputs = pipe(
        all_messages, 
        max_new_tokens=1024, 
        do_sample=False, 
        batch_size=2,
        truncation=True
    )

    final_results = []
    for i, output in enumerate(outputs):
        # Extract the assistant's last response
        llm_response = output[0]["generated_text"][-1]["content"].strip()
        final_results.append({
            "user_id": prompts_list[i]["user_id"],
            "llm_ranking": llm_response
        })
    
    return final_results

def main():
    input_file = 'enriched_user_profiles.json'
    #input_file = 'test.json'
    #output_file = 'test_ranking.json'

    # Switch between these to test different models:
    #MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
    #MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
    MODEL_NAME = "google/gemma-3-4b-it"

    if MODEL_NAME == "meta-llama/Llama-3.2-3B-Instruct":
        model_file_name = "llama"
    elif MODEL_NAME == "mistralai/Mistral-7B-Instruct-v0.3":
        model_file_name = "mistral"
    elif MODEL_NAME == "google/gemma-3-4b-it":
        model_file_name = "gemma"
    else:
        raise ValueError(f"Unknown MODEL_NAME: {MODEL_NAME}")


    output_file = f'with_stem_and_emotion/{model_file_name}_user_rankings_results.json'

    # --- EXECUTION ---
    try:
        start_time = time.perf_counter()
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 1. Prepare text prompts
        prompts = generate_user_prompts(data)

        # 2. Run LLM on GPU
        results = run_prompts(prompts, MODEL_NAME)

        # 3. Save output
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)

        end_time = time.perf_counter()
        print(f"Execution time: {end_time - start_time:.6f} seconds")
        print(f"\nSUCCESS: Results saved to {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Could not find {input_file}")
    except Exception as e:
        print(f"An error occurred: {e}")


        

if __name__ == "__main__":
    main()