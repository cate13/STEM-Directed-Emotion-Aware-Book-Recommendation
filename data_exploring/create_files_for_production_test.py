import random
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_VECTOR = os.path.join(
    BASE_DIR, "processed_data", "books_with_subjects_read_by_younger_readers.jsonl"
)

def sample_jsonl(input_filename):
    # Step 1: Read all lines from the source file
    print("Reading source file...")
    with open(input_filename, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    
    total_lines = len(all_lines)
    print(f"Loaded {total_lines} lines.")
    
    # Quick safety check
    if total_lines < 1100:
        print("Warning: The source file has very few lines for this sampling.")

    # Step 2: Define the sampling requirements
    # File 1 needs 1000 lines. Files 2-10 need a random count between 3 and 10.
    sample_sizes = [1000] + [random.randint(3, 10) for _ in range(9)]
    total_needed = sum(sample_sizes)
    
    # Step 3: Randomly sample the required number of total lines all at once
    # This ensures no duplicate lines across any of the 10 files
    sampled_lines = random.sample(all_lines, total_needed)
    
    # Step 4: Distribute the sampled lines into the 10 files
    current_index = 0
    for i, size in enumerate(sample_sizes, start=1):
        output_filename = os.path.join(
            BASE_DIR, "Production", "test_files", f"sampled_output_{i}_{size}.jsonl"
        )
        
        # Extract the specific slice for this file
        file_lines = sampled_lines[current_index : current_index + size]
        current_index += size
        
        # Write to the new jsonl file
        with open(output_filename, 'w', encoding='utf-8') as out_f:
            out_f.writelines(file_lines)
            
        print(f"Created {output_filename} with {size} lines.")

sample_jsonl(BASE_VECTOR)