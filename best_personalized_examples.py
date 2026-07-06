import json
from sklearn.metrics import ndcg_score

with open("recommendations/no_stem_test/with_personalized.json") as f:
    A = json.load(f)

with open("recommendations/no_stem_test/age_based.json") as f:
    B = json.load(f)

def load_book_lookup(jsonl_path):
    lookup = {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            book = json.loads(line)

            isbn = str(book["ISBN"]).strip()

            lookup[isbn] = {
                "title": book.get("Book-Title", "Unknown Title"),
                "author": book.get("Book-Author", "Unknown Author"),
            }

    return lookup

def create_comparison_report(
    user_ids,
    A,
    B,
    books_jsonl,
    output_file="comparison_report.txt"
):
    book_lookup = load_book_lookup(books_jsonl)

    A_users = {u["user_id"]: u for u in A}
    B_users = {u["user_id"]: u for u in B}

    with open(output_file, "w", encoding="utf-8") as out:

        for user_id in user_ids:

            out.write("=" * 100 + "\n")
            out.write(f"USER {user_id}\n")
            out.write("=" * 100 + "\n\n")

            recs_a = sorted(
                A_users[user_id]["recommendation_list"],
                key=lambda x: x["cos"],
                reverse=True,
            )

            recs_b = sorted(
                B_users[user_id]["recommendation_list"],
                key=lambda x: x["cos"],
                reverse=True,
            )

            out.write("PERSONALIZED ORDER\n")
            out.write("-" * 40 + "\n")

            for idx, rec in enumerate(recs_a, start=1):
                isbn = str(rec["isbn"]).strip()

                book = book_lookup.get(
                    isbn,
                    {
                        "title": f"Unknown ISBN {isbn}",
                        "author": "Unknown Author",
                    },
                )

                out.write(
                    f"{idx}. {book['title']} by {book['author']} "
                    f"(ground truth rating={rec['rating']}, "
                    f"cos={rec['cos']:.4f})\n"
                )

            out.write("\n")

            out.write("AGE BASED ORDER\n")
            out.write("-" * 40 + "\n")

            for idx, rec in enumerate(recs_b, start=1):
                isbn = str(rec["isbn"]).strip()

                book = book_lookup.get(
                    isbn,
                    {
                        "title": f"Unknown ISBN {isbn}",
                        "author": "Unknown Author",
                    },
                )

                out.write(
                    f"{idx}. {book['title']} by {book['author']} "
                    f"(ground truth rating={rec['rating']}, "
                    f"cos={rec['cos']:.4f})\n"
                )

            out.write("\n\n")

    print(f"Saved report to {output_file}")

def get_most_diff():
    A_users = {u["user_id"]: u for u in A}
    B_users = {u["user_id"]: u for u in B}

    results = []

    for user_id in A_users:

        recs_a = A_users[user_id]["recommendation_list"]
        recs_b = B_users[user_id]["recommendation_list"]

        ndcg_a = ndcg_score(
            [[r["rating"] for r in recs_a]],
            [[r["cos"] for r in recs_a]]
        )

        ndcg_b = ndcg_score(
            [[r["rating"] for r in recs_b]],
            [[r["cos"] for r in recs_b]]
        )

        diff = ndcg_a - ndcg_b

        if diff > 0:
            results.append({
                "user_id": user_id,
                "ndcg_a": ndcg_a,
                "ndcg_b": ndcg_b,
                "diff": diff
            })

    results.sort(key=lambda x: x["diff"], reverse=True)

    results_list = []
    for r in results[:10]:
        print(
            f"user={r['user_id']}, "
            f"A={r['ndcg_a']:.4f}, "
            f"B={r['ndcg_b']:.4f}, "
            f"diff={r['diff']:.4f}"
        )
        results_list.append(r['user_id'])
    return results_list
    

top_users = get_most_diff()

create_comparison_report(
    user_ids=top_users,
    A=A,
    B=B,
    books_jsonl="processed_data/books_with_subjects_read_by_younger_readers.jsonl",
    output_file="top10_user_comparison.txt",
)