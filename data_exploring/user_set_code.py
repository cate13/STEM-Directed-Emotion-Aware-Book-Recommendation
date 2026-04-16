import json
import os

def filter_users():
    txt_path = 'users_with_one_STEM_book.txt'
    jsonl_source_path = '../processed_data/curated_users.jsonl'
    output_path = 'filtered_stem_users.jsonl'

    with open(txt_path, 'r') as f:
        target_users = {int(line.strip()) for line in f if line.strip()}

    count = 0
    
    with open(jsonl_source_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            user_data = json.loads(line)
            
            if user_data.get('user_id') in target_users:
                outfile.write(json.dumps(user_data) + '\n')
                count += 1

    print(f"Done! Created '{output_path}' with {count} users.")

if __name__ == "__main__":
    filter_users()