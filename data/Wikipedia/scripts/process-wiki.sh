python scripts/build_db.py --data_path ./wiki-pages --save_path ./data/wikipedia.db
python scripts/build_tfidf.py --db_path ./data/wikipedia.db --out_path ./data/tfidf.npz