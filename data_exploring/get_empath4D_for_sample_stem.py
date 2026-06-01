import json

from Vectorizer.EmpathVectorMaker import EmpathVectorMaker

vector_maker = EmpathVectorMaker()

def process_descriptions(input_path, output_path):
    processed_count = 0

    # Open the input file to read and the output file to write simultaneously
    with open(input_path, "r", encoding="utf-8") as infile, open(
        output_path, "w", encoding="utf-8"
    ) as outfile:

        for line_num, line in enumerate(infile, 1):
            try:
                # 1. Parse the current line
                book_data = json.loads(line)
                isbn = book_data.get("ISBN")
                description = book_data.get("description", "")

                # 2. Run the description through your custom method
                empath4d = vector_maker.getEmapthVector(description)

                # 3. Create the new dictionary structure
                output_data = {
                    "ISBN": isbn,
                    "description": description,
                    "empath4D": empath4d,  #
                }

                # 4. Write it immediately to the new JSONL file
                outfile.write(json.dumps(output_data, ensure_ascii=False) + "\n")
                processed_count += 1

            except json.JSONDecodeError:
                print(f"Skipping malformed JSON data on line {line_num}")
                continue

    print(
        f"Finished! Successfully processed {processed_count} lines and saved to {output_path}."
    )


if __name__ == "__main__":
    process_descriptions("sample_stem.jsonl", "sample_stem_with_empath_4d.jsonl")