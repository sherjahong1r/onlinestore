import os # Operatsion tizim bilan ishlash (fayllar va muhit o'zgaruvchilari).
import psycopg2 # PostgreSQL bazasiga Python orqali ulanish uchun asosiy kutubxona.
from dotenv import load_dotenv # .env faylidan bazaga ulanish paroli va URL manzilini xavfsiz o'qib olish.

load_dotenv()
DB_URL = os.getenv("DB_URL")

def get_db_schema() -> str: # Bu tushuntirishlar AI uchun yo'riqnoma vazifasini o'taydi.
    return """
    SEN FAQAT QUYIDAGI 26 TA JADVALDAN FOYDALANISHINGIZ KERAK!
    Boshqa hech qanday jadval (information_schema, pg_catalog va boshqalar) ISHLATMA!

    MAVJUD JADVALLAR (global_data bazasi):

    1.  companies
        - id, name, industry, foundation_year, headquarters_city

    2.  countries
        - id, name, region, population, gdp_per_capita

    3.  cities
        - id, country_id, name, is_capital

    4.  revenue_data
        - id, company_id, year, total_revenue_billion_usd, currency

    5.  investment_rounds
        - id, company_id, round_name, amount_raised_million_usd, year

    6.  brand_valuation
        - id, company_id, brand_value_billion_usd, valuation_year, ranking_global

    7.  employees
        - id, company_id, total_headcount, avg_salary_usd, remote_percentage

    8.  executives
        - id, company_id, full_name, position, years_in_role

    9.  corporate_governance
        - id, company_id, board_size, independent_directors, governance_rating

    10. competitors
        - id, company_id, competitor_name, market_share_impact

    11. customer_reviews
        - id, company_id, review_rating, review_count, sentiment_score

    12. customer_churn_rate
        - id, company_id, churn_rate_pct, segment, quarter

    13. industry_trends
        - id, industry_name, growth_rate_pct, year, top_technology

    14. marketing_campaigns
        - id, company_id, campaign_name, budget_million_usd, start_year

    15. patents
        - id, company_id, title, patent_number, registration_year

    16. technologies
        - id, name, category, open_source, creator_company_id

    17. intellectual_property_licensing
        - id, patent_id, licensee_company_id, royalty_rate_pct, contract_status

    18. products
        - id, company_id, name, category, release_year

    19. supply_chain
        - id, company_id, region, warehouse_count, logistics_partner

    20. infrastructure_projects
        - id, company_id, project_name, budget_million_usd, completion_status

    21. global_offices
        - id, company_id, country_id, office_size_sqm, is_headquarters

    22. partnerships
        - id, company_id_1, company_id_2, partnership_type, active

    23. legal_issues
        - id, company_id, case_title, status, penalty_amount_million_usd

    24. sustainability_reports
        - id, company_id, esg_score, carbon_footprint_tons, renewable_energy_pct

    25. investors
        - id, name, type, headquarters

    26. data_dictionary
        - id, table_name, category, description, purpose

    MUHIM ESLATMALAR:
    - Kompaniya nomi = companies.name
    - Kompaniya sektori = companies.industry
    - Barcha jadvallar company_id orqali companies.id ga bog'langan
    - Kompaniya nomi kerak bo'lsa JOIN ishlatish:
      JOIN companies c ON jadval.company_id = c.id
    """

def execute_sql(query: str):  # Bu funksiya AI yozgan SQL kodni haqiqatda bazaga yuboradi:
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()
        # Baza bilan aloqa o'rnatadi (connect) va buyruqlarni yuborish uchun "kursor" yaratadi.
        cursor.execute(query) # AI yozib bergan SQL so'rovini (query) bazaga yuboradi.
        if query.strip().upper().startswith("SELECT"):
            result = cursor.fetchall()
        else:
            conn.commit()
            result = "Muvaffaqiyatli bajarildi."
        cursor.close()
        conn.close() # Ish tugagach, xotirani band qilmaslik uchun bazaga ulanishni yopadi. 
        return result
    except Exception as e:
        return f"SQL Xatolik: {str(e)}"


