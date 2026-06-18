import os
import re
import psycopg # Python va PostgreSQL o‘rtasidagi bog‘lovchi
from psycopg.rows import dict_row # bazadan kelgan ma’lumotlarni oddiy "ro‘yxat" emas, balki "lug‘at" (kalit-qiymat) ko‘rinishida olish uchun.
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DB_URL")

if not DB_URL:
    raise RuntimeError(
        "DB_URL topilmadi. .env faylida DB_URL=postgresql://user:parol@host:port/baza "
        "ko'rinishida belgilang."
    )


def _get_connection(): # Har safar bazaga kirganda yangi ulanish ochadi
    """Yangi PostgreSQL ulanish ochadi."""
    return psycopg.connect(DB_URL, row_factory=dict_row)


def get_db_schema() -> str:
    """
    Bazadagi barcha jadvallar va ularning ustunlarini LLM uchun
    o'qilishi oson bo'lgan matn ko'rinishida qaytaradi.

    Masalan:
        Jadval: employees
          - id (integer)
          - full_name (text)
          - salary (numeric)

        Jadval: companies
          - id (integer)
          - name (text)
    """
    query = """
        SELECT
            t.table_name,
            c.column_name,
            c.data_type
        FROM information_schema.tables t
        JOIN information_schema.columns c
            ON t.table_name = c.table_name
        WHERE t.table_schema = 'public'
          AND t.table_type = 'BASE TABLE'
        ORDER BY t.table_name, c.ordinal_position;
    """
    try:
        with _get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                rows = cur.fetchall()
    except Exception as e:
        return f"[XATO] Sxema o'qib bo'lmadi: {e}"

    if not rows:
        return "Bazada jadval topilmadi."

    schema_by_table = {}
    for row in rows:
        table = row["table_name"]
        schema_by_table.setdefault(table, []).append(
            f"  - {row['column_name']} ({row['data_type']})"
        )

    parts = []
    for table, columns in schema_by_table.items():
        parts.append(f"Jadval: {table}\n" + "\n".join(columns))

    return "\n\n".join(parts)

def get_table_list() -> list[str]:
    """Bazadagi barcha jadval nomlarini ro'yxat ko'rinishida qaytaradi."""
    query = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        ORDER BY table_name;
    """
    try:
        with _get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                return [r["table_name"] for r in cur.fetchall()]
    except Exception:
        return []


# get_table_names_from_sql: AI yozgan SQL so‘rovini tahlil qilib, u qaysi jadvalga murojaat
# qilayotganini aniqlaydi. Bu keyinchalik javobda "Manba: employees jadvali" deb yozish uchun kerak.
def get_table_names_from_sql(sql: str) -> list[str]:
    """
    Berilgan SQL so'rov matnidan qaysi jadval(lar) ishlatilganini aniqlaydi.
    Bu manba sifatida foydalanuvchiga "qaysi jadvaldan olindi" deb
    ko'rsatish uchun ishlatiladi.
    """
    known_tables = get_table_list()
    used = []
    sql_lower = sql.lower()
    for table in known_tables:
        # so'z chegarasi bilan qidiramiz, masalan "products" so'zi
        # "products_old" ichida ham topilmasligi uchun \b ishlatamiz
        if re.search(rf"\b{re.escape(table.lower())}\b", sql_lower):
            used.append(table)
    return used


def is_safe_select(sql: str) -> bool:
    """
    SQL so'rovning faqat SELECT (yoki WITH ... SELECT) ekanligini tekshiradi.
    INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE kabi buyruqlarga ruxsat berilmaydi.
    """
    cleaned = sql.strip().lower()
    if not (cleaned.startswith("select") or cleaned.startswith("with")):
        return False
    dangerous_keywords = [
        "insert", "update", "delete", "drop", "alter",
        "truncate", "create", "grant", "revoke",
    ]
    # WITH ... SELECT ichida ham xavfli so'z bo'lmasligini tekshiramiz
    for word in dangerous_keywords:
        if re.search(rf"\b{word}\b", cleaned):
            return False
    return True


def get_vector_enabled_tables() -> dict:
    """
    Bazada `vector` turidagi ustunga ega bo'lgan jadvallarni topadi.

    Qaytadi:
        {
            "products": {"embedding_column": "embedding", "text_columns": [...]},
            "company_embeddings": {...},
            ...
        }

    text_columns - vector bilan birga saqlangan, foydalanuvchiga ko'rsatish
    uchun mazmunli bo'lgan ustunlar (TEXT/VARCHAR turidagilar).
    """
    query = """
        SELECT
            c.table_name,
            c.column_name,
            c.udt_name
        FROM information_schema.columns c
        WHERE c.table_schema = 'public'
        ORDER BY c.table_name, c.ordinal_position;
    """
    try:
        with _get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                rows = cur.fetchall()
    except Exception:
        return {}

    tables = {}
    for row in rows:
        tables.setdefault(row["table_name"], []).append(row)

    result = {}
    for table_name, columns in tables.items():
        embedding_col = None
        text_cols = []
        for col in columns:
            if col["udt_name"] == "vector":
                embedding_col = col["column_name"]
            elif col["udt_name"] in ("text", "varchar"):
                text_cols.append(col["column_name"])
        if embedding_col:
            result[table_name] = {
                "embedding_column": embedding_col,
                "text_columns": text_cols,
            }
    return result

# Misol: Foydalanuvchi "arzonroq telefonlar" deb yozsa, bu qism matn ma’nosiga eng yaqin bo‘lgan qatorlarni topib beradi.
def search_vector_db(query_text: str, embed_fn, table: str = None, top_k: int = 5):
    """
    Berilgan matnga semantik jihatdan eng o'xshash qatorlarni topadi.

    Parametrlar:
        query_text : foydalanuvchi savoli / qidiruv matni
        embed_fn   : matnni vectorga aylantiruvchi funksiya (masalan
                     SentenceTransformer.encode), tashqaridan beriladi -
                     shunda bu modul embedding modeliga bog'liq bo'lmaydi.
        table      : agar ko'rsatilsa, faqat shu jadvalda qidiriladi.
                     Ko'rsatilmasa, vector ustuniga ega birinchi jadval
                     ishlatiladi.
        top_k      : nechta natija qaytarilishi

    Qaytadi:
        {
            "success": bool,
            "rows": list[dict] | None,
            "error": str | None,
            "table_used": str | None,
        }
    """
    vector_tables = get_vector_enabled_tables()

    if not vector_tables:
        return {
            "success": False,
            "rows": None,
            "error": "Bazada vector ustuniga ega jadval topilmadi.",
            "table_used": None,
        }

    if table and table in vector_tables:
        target_table = table
    else:
        target_table = next(iter(vector_tables))

    info = vector_tables[target_table]
    embedding_col = info["embedding_column"]
    text_cols = info["text_columns"]

    select_cols = ", ".join(["id"] + text_cols) if text_cols else "*"

    try:
        query_vector = embed_fn(query_text)
        # psycopg uchun vector'ni PostgreSQL formatiga ([1,2,3]) o'giramiz
        vector_literal = "[" + ",".join(str(float(x)) for x in query_vector) + "]"

        sql = f"""
            SELECT {select_cols},
                   {embedding_col} <-> %s::vector AS distance
            FROM {target_table}
            WHERE {embedding_col} IS NOT NULL
            ORDER BY {embedding_col} <-> %s::vector
            LIMIT %s;
        """

        with _get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (vector_literal, vector_literal, top_k))
                rows = cur.fetchall()

        return {
            "success": True,
            "rows": rows,
            "error": None,
            "table_used": target_table,
        }
    except Exception as e:
        return {
            "success": False,
            "rows": None,
            "error": str(e),
            "table_used": target_table,
        }

# Vazifasi: Tekshiruvdan o‘tgan SQL so‘rovini bazaga yuboradi, natijani kutib oladi va xatolik bo‘lsa,
# uni chiroyli tarzda (xato matni bilan) qaytaradi.
def execute_sql(sql: str):
    """
    SQL so'rovni bazada bajaradi va natijani qaytaradi.

    Qaytadi:
        dict ko'rinishida:
            {
                "success": bool,
                "rows": list[dict] | None,
                "error": str | None,
                "tables_used": list[str],
            }
    """
    sql = sql.strip().rstrip(";")

    tables_used = get_table_names_from_sql(sql)

    if not is_safe_select(sql):
        return {
            "success": False,
            "rows": None,
            "error": "Xavfsizlik sababli faqat SELECT so'rovlarga ruxsat berilgan.",
            "tables_used": tables_used,
        }

    try:
        with _get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
                return {
                    "success": True,
                    "rows": rows,
                    "error": None,
                    "tables_used": tables_used,
                }
    except Exception as e:
        return {
            "success": False,
            "rows": None,
            "error": str(e),
            "tables_used": tables_used,
        }
    
# Bu fayl "filtir" va "o‘tkazgich" vazifasini bajaradi:
# Bazani "o‘qiydi" (schema).
# AI so‘rovlarini "tekshiradi" (xavfsizlik).
# AIga "vektorli" (aqlli) qidiruv imkonini beradi.
# Natijani AIga tushunarli formatda qaytaradi.


