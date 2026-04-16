import json
from tqdm import tqdm
import time
import requests
import xml.etree.ElementTree as ET
from typing import List, Optional
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

INDEX = 0

###############################################################################
# 1. LOC (Library of Congress) SUBJECT HEADING FUNCTIONS
###############################################################################

def fetch_loc_mods_xml(isbn: str) -> Optional[ET.Element]:
    isbn = isbn.strip()
    if not isbn:
        return None

    url = (
        "http://lx2.loc.gov:210/lcdb"
        "?version=1.1"
        "&operation=searchRetrieve"
        f"&query=bath.isbn={isbn}"
        "&maximumRecords=1"
        "&recordSchema=mods"
    )

    for attempt in range(1, 4):
        try:
            response = requests.get(
                url,
                timeout=15,
                headers={"Connection": "close", "User-Agent": "loc-subject-script"},
            )
            break
        except requests.exceptions.RequestException:
            if attempt == 3:
                return None
            time.sleep(5 * attempt)

    if response.status_code != 200:
        return None

    try:
        root = ET.fromstring(response.text)
    except ET.ParseError:
        return None

    time.sleep(2)
    return root

def extract_subject_headings_from_mods(root: ET.Element) -> List[str]:
    ns = {
        "mods": "http://www.loc.gov/mods/v3",
    }

    subject_headings = []

    for mods_record in root.findall(".//mods:mods", ns):
        for subj in mods_record.findall("mods:subject", ns):
            parts = []

            for tag in ("topic", "geographic", "genre"):
                for child in subj.findall(f"mods:{tag}", ns):
                    text = (child.text or "").strip()
                    if text:
                        parts.append(text)

            if parts:
                heading = " -- ".join(parts)
                if heading not in subject_headings:
                    subject_headings.append(heading)

    return subject_headings


def get_subject_headings_for_isbn(isbn: str) -> List[str]:
    root = fetch_loc_mods_xml(isbn)
    if root is None:
        return []
    return extract_subject_headings_from_mods(root)

###############################################################################
# 2. GOOGLE BOOKS WITH RATE-LIMIT HANDLING
###############################################################################

def get_google_books_metadata(isbn: str, max_retries=5, base_delay=1.0):
    """
    Returns: (description, categories)
    description -> str or None
    categories  -> list[str] or empty list
    """
    url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}"

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, timeout=10)
            status = resp.status_code

            if status != 200:
                delay = base_delay * (2 ** (attempt - 1))
                time.sleep(delay)
                continue

            try:
                data = resp.json()
            except ValueError:
                delay = base_delay * (2 ** (attempt - 1))
                time.sleep(delay)
                continue

            items = data.get("items", [])
            if not items:
                return None, []

            info = items[0].get("volumeInfo", {})

            description = info.get("description")
            categories = info.get("categories", [])

            if categories is None:
                categories = []

            return description, categories

        except requests.exceptions.RequestException:
            delay = base_delay * (2 ** (attempt - 1))
            time.sleep(delay)

    return None, []

def is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except:
        return False

###############################################################################
# 3. MAIN PROCESSOR FOR CSV → JSONL
###############################################################################

def process_books_txt_to_jsonl(
    input_txt_path: str,
    jsonl_output_path: str,
    missing_desc_path: str = "valid_isbn_missing_cat.txt",
    non_english_path: str = "not_english.txt",
):

    try:
        with open(input_txt_path, newline="", encoding="utf-8") as infile, \
            open(jsonl_output_path, "a", encoding="utf-8") as jsonl_out, \
            open(missing_desc_path, "a", encoding="utf-8") as f_missing, \
            open(non_english_path, "a", encoding="utf-8") as f_noneng:

            i = 0
            for line in tqdm(infile):
                i += 1
                isbn = line.strip()
                
                if not isbn:
                    continue

                # --- GET DESCRIPTION ---
                desc, categories = get_google_books_metadata(isbn)

                if not desc:
                    f_missing.write(isbn + "\n")
                    continue

                if not is_english(desc):
                    f_noneng.write(isbn + "\n")
                    continue

                subjects = get_subject_headings_for_isbn(isbn)

                record = {
                    "ISBN": isbn,
                    "description": desc,
                    "LoC_subjects": subjects,
                    "Google_categories": categories
                }

                jsonl_out.write(json.dumps(record, ensure_ascii=False) + "\n")

                if i % 100 == 0:
                    print(isbn)
    except Exception as e:
        print(e)
    

###############################################################################
# 4. CALL FUNCTION
###############################################################################

if __name__ == "__main__":
    process_books_txt_to_jsonl(
        input_txt_path="second_try_isbns.txt",
        jsonl_output_path="data/books_with_subjects_2.jsonl",
    )
