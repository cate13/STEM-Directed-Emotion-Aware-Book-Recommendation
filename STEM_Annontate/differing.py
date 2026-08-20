import csv

input_file = "STEM_Annontate/Annotated Books - 400.csv"  # Replace with your CSV file path
output_file = "STEM_Annontate/disagreeing_isbns.txt"

with open(input_file, mode="r", encoding="utf-8") as infile, open(
    output_file, mode="w", encoding="utf-8"
) as outfile:

    reader = csv.DictReader(infile)

    for row in reader:
        # Compare user_1 and user_2 column values
        if row["user_1"].strip() != row["user_2"].strip():
            outfile.write(f"{row['isbn']}\n")