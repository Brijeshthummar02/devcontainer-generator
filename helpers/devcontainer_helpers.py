import sqlite3
import numpy as np
from helpers.openai_helpers import get_embedding
from helpers.jinja_helper import process_template

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def fetch_similar_configs_from_embeddings(user_description, top_k=3):
    logging.info("Fetching similar devcontainer configs using embeddings...")

    user_embedding = get_embedding(user_description)
    conn = sqlite3.connect("data/devcontainers.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, embedding, config FROM devcontainers")
    rows = cursor.fetchall()
    conn.close()

    candidates = []
    for row in rows:
        try:
            row_embedding = eval(row[1])
            score = cosine_similarity(user_embedding, row_embedding)
            candidates.append((score, row[2]))  # row[2] = JSON string config
        except Exception as e:
            logging.warning(f"Skipping row due to error: {e}")

    top_matches = sorted(candidates, reverse=True)[:top_k]
    return [eval(config) for _, config in top_matches]

def generate_devcontainer_from_description(description):
    logging.info("Generating devcontainer.json from user-provided description...")

    configs = fetch_similar_configs_from_embeddings(description)
    if not configs:
        raise ValueError("No similar configurations found.")

    # Merge top configs (you can make this smarter if needed)
    final_config = {}
    for config in configs:
        final_config.update({k: v for k, v in config.items() if k not in final_config})

    rendered = process_template("prompts/devcontainer.jinja", {"config": final_config})
    
    os.makedirs(".devcontainer", exist_ok=True)
    with open(".devcontainer/devcontainer.json", "w") as f:
        f.write(rendered)

    logging.info("devcontainer.json successfully generated and written.")
    return rendered
