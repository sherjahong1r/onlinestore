import psycopg2
from sentence_transformers import SentenceTransformer

# 1. Sozlamalar
DB_URL = "postgresql://postgres:jahongir@localhost:5433/global_data"
model = SentenceTransformer("all-MiniLM-L6-v2")

def auto_embed():
    
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    # Bazadagi barcha jadvallarni olish
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
    tables = [t[0] for t in cur.fetchall()]

    for table in tables:
        # Tizim jadvallarini o'tkazib yuboramiz
        if table in ['checkpoint_blobs', 'checkpoints', 'checkpoint_migrations', 'checkpoint_writes']:
            continue
            
        print(f"--- {table} jadvali tekshirilmoqda ---")
        
        # Ustunlarni aniqlash (text yoki varchar bo'lganlarini)
        cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table}' AND data_type IN ('text', 'character varying');")
        text_cols = [c[0] for c in cur.fetchall()]
        
        if not text_cols: continue
        
        # Vektorlash
        target_col = text_cols[0] # Birinchi matnli ustunni olamiz
        cur.execute(f"SELECT id, {target_col} FROM {table} WHERE embedding IS NULL AND {target_col} IS NOT NULL;")
        rows = cur.fetchall()
        
        for row in rows:
            vec = model.encode(str(row[1])).tolist()
            cur.execute(f"UPDATE {table} SET embedding = %s WHERE id = %s;", (vec, row[0]))
        
        conn.commit()
        print(f"{table} jadvali vektorlandi.")

    cur.close()
    conn.close()

if __name__ == "__main__":
    auto_embed()