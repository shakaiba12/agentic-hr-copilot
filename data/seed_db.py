"""
PeopleQuery AI - HR Relational Database Seeding Script
Populates SQLite / PostgreSQL database with realistic HR workforce data.
"""
import os
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "hr_database.sqlite"
SCHEMA_PATH = BASE_DIR / "hr_schema.sql"


def seed_database(db_path: Path = DB_PATH, schema_path: Path = SCHEMA_PATH) -> None:
    """Initialize and populate the SQLite database with rich HR test records."""
    if db_path.exists():
        os.remove(db_path)

    conn = sqlite3.connect(db_path)

    # 1. Execute Schema
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()
    conn.executescript(schema_sql)

    # 2. Seed Departments
    departments = [
        (1, "Engineering", "ENG", "San Francisco, CA", 4500000.00),
        (2, "Sales", "SAL", "New York, NY", 3200000.00),
        (3, "Marketing", "MKT", "San Francisco, CA", 1800000.00),
        (4, "People Operations", "HR", "Austin, TX", 950000.00),
        (5, "Finance", "FIN", "New York, NY", 1200000.00),
        (6, "Customer Support", "SUP", "Austin, TX", 850000.00),
    ]
    conn.executemany(
        "INSERT INTO departments (id, name, code, location, budget) VALUES (?, ?, ?, ?, ?)",
        departments,
    )

    # 3. Seed Positions
    positions = [
        (1, "VP of Engineering", 1, 220000.00, 300000.00),
        (2, "Engineering Manager", 1, 170000.00, 230000.00),
        (3, "Senior Software Engineer", 1, 140000.00, 190000.00),
        (4, "Software Engineer", 1, 105000.00, 145000.00),
        (5, "VP of Sales", 2, 200000.00, 280000.00),
        (6, "Senior Account Executive", 2, 110000.00, 160000.00),
        (7, "Account Executive", 2, 75000.00, 115000.00),
        (8, "Head of Marketing", 3, 160000.00, 220000.00),
        (9, "Growth Marketing Specialist", 3, 85000.00, 125000.00),
        (10, "Head of People", 4, 160000.00, 210000.00),
        (11, "HR Business Partner", 4, 90000.00, 130000.00),
        (12, "Talent Acquisition Lead", 4, 95000.00, 135000.00),
        (13, "Director of Finance", 5, 170000.00, 230000.00),
        (14, "Financial Analyst", 5, 85000.00, 120000.00),
        (15, "Support Lead", 6, 80000.00, 110000.00),
        (16, "Support Specialist", 6, 55000.00, 75000.00),
    ]
    conn.executemany(
        "INSERT INTO positions (id, title, department_id, min_salary, max_salary) VALUES (?, ?, ?, ?, ?)",
        positions,
    )

    # 4. Seed Employees
    # Notice varying hire dates (some > 12 months tenure, some recent hires)
    employees = [
        # Executives & Managers
        (1, "Elena", "Rostova", "elena.rostova@example.com", "Female", 1, 1, "2021-03-15", "ACTIVE", "FULL_TIME", 260000.00, 4.8, None),
        (2, "Marcus", "Chen", "marcus.chen@example.com", "Male", 2, 5, "2021-06-01", "ACTIVE", "FULL_TIME", 240000.00, 4.6, None),
        (3, "Sarah", "Jenkins", "sarah.jenkins@example.com", "Female", 4, 10, "2022-01-10", "ACTIVE", "FULL_TIME", 185000.00, 4.7, None),
        (4, "David", "Kim", "david.kim@example.com", "Male", 5, 13, "2022-04-01", "ACTIVE", "FULL_TIME", 195000.00, 4.5, None),
        (5, "Maya", "Patel", "maya.patel@example.com", "Female", 3, 8, "2022-08-15", "ACTIVE", "FULL_TIME", 175000.00, 4.4, None),
        # Engineering
        (6, "Alex", "Mercer", "alex.mercer@example.com", "Male", 1, 2, "2022-02-01", "ACTIVE", "FULL_TIME", 185000.00, 4.5, 1),
        (7, "Priya", "Sharma", "priya.sharma@example.com", "Female", 1, 3, "2023-01-15", "ACTIVE", "FULL_TIME", 165000.00, 4.9, 6),
        (8, "Liam", "O'Connor", "liam.oconnor@example.com", "Male", 1, 3, "2023-05-10", "ACTIVE", "FULL_TIME", 155000.00, 4.1, 6),
        (9, "Sofia", "Rodriguez", "sofia.rodriguez@example.com", "Female", 1, 4, "2024-02-20", "ACTIVE", "FULL_TIME", 125000.00, 4.2, 6),
        (10, "Lucas", "Vanderbilt", "lucas.vanderbilt@example.com", "Male", 1, 4, "2024-09-01", "ACTIVE", "FULL_TIME", 115000.00, 3.8, 6),
        (11, "Amina", "Diallo", "amina.diallo@example.com", "Female", 1, 4, "2025-11-15", "ACTIVE", "FULL_TIME", 110000.00, 3.5, 6), # Recent hire (<1 yr)
        # Sales
        (12, "James", "Wilson", "james.wilson@example.com", "Male", 2, 6, "2022-09-12", "ACTIVE", "FULL_TIME", 145000.00, 4.3, 2),
        (13, "Chloe", "Bennett", "chloe.bennett@example.com", "Female", 2, 6, "2023-03-01", "ACTIVE", "FULL_TIME", 140000.00, 4.6, 2),
        (14, "Daniel", "Foster", "daniel.foster@example.com", "Male", 2, 7, "2024-01-15", "ACTIVE", "FULL_TIME", 95000.00, 3.9, 2),
        (15, "Grace", "Hopper", "grace.hopper@example.com", "Female", 2, 7, "2025-06-01", "ACTIVE", "FULL_TIME", 85000.00, 3.7, 2),
        # Marketing
        (16, "Ethan", "Hunt", "ethan.hunt@example.com", "Male", 3, 9, "2023-07-20", "ACTIVE", "FULL_TIME", 105000.00, 4.0, 5),
        (17, "Zoe", "Kravitz", "zoe.kravitz@example.com", "Female", 3, 9, "2024-10-01", "ACTIVE", "FULL_TIME", 98000.00, 4.1, 5),
        # People Ops (HR)
        (18, "Rachel", "Green", "rachel.green@example.com", "Female", 4, 11, "2022-11-01", "ACTIVE", "FULL_TIME", 115000.00, 4.6, 3),
        (19, "Hannah", "Abbott", "hannah.abbott@example.com", "Female", 4, 12, "2023-08-15", "ACTIVE", "FULL_TIME", 108000.00, 4.4, 3),
        # Finance
        (20, "Oliver", "Queen", "oliver.queen@example.com", "Male", 5, 14, "2023-02-01", "ACTIVE", "FULL_TIME", 105000.00, 4.2, 4),
        (21, "Emily", "Thorne", "emily.thorne@example.com", "Female", 5, 14, "2024-04-15", "ACTIVE", "FULL_TIME", 95000.00, 4.3, 4),
        # Customer Support
        (22, "Sam", "Winchester", "sam.winchester@example.com", "Male", 6, 15, "2022-05-10", "ACTIVE", "FULL_TIME", 90000.00, 4.1, None),
        (23, "Dean", "Winchester", "dean.winchester@example.com", "Male", 6, 16, "2023-10-01", "ACTIVE", "FULL_TIME", 68000.00, 3.9, 22),
        (24, "Clara", "Oswald", "clara.oswald@example.com", "Female", 6, 16, "2024-08-01", "ON_LEAVE", "FULL_TIME", 65000.00, 4.0, 22),
        (25, "Ruby", "Sunday", "ruby.sunday@example.com", "Female", 6, 16, "2026-01-10", "ACTIVE", "FULL_TIME", 62000.00, 3.2, 22),
    ]
    conn.executemany(
        """
        INSERT INTO employees (
            id, first_name, last_name, email, gender, department_id, position_id,
            hire_date, employment_status, employment_type, salary, performance_rating, manager_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        employees,
    )

    # 5. Seed Benefits
    benefits = [
        (1, "Comprehensive Health & Dental", "HEALTH_PREMIUM", "100% employee health and dental coverage", 0),
        (2, "401(k) Retirement Match", "401K_MATCH", "50% match up to 6% of salary", 6),
        (3, "Tuition & Education Assistance", "TUITION_AID", "Up to $3000/yr for accredited degree/certs", 12),
        (4, "One-Time Home Office Stipend", "REMOTE_STIPEND", "$500 one-time ergonomic stipend", 3),
        (5, "Maternity Leave Program", "MATERNITY_LEAVE", "16 weeks fully paid leave", 12),
    ]
    conn.executemany(
        "INSERT INTO benefits (id, name, code, description, min_tenure_months) VALUES (?, ?, ?, ?, ?)",
        benefits,
    )

    # 6. Seed Leaves
    leaves = [
        (1, 24, "MATERNITY", "2026-02-01", "2026-05-24", 80, "APPROVED"),
        (2, 7, "ANNUAL", "2025-12-20", "2025-12-31", 8, "APPROVED"),
        (3, 8, "SICK", "2026-01-15", "2026-01-16", 2, "APPROVED"),
        (4, 13, "ANNUAL", "2025-07-10", "2025-07-20", 8, "APPROVED"),
        (5, 18, "ANNUAL", "2025-08-01", "2025-08-10", 7, "APPROVED"),
    ]
    conn.executemany(
        "INSERT INTO leaves (id, employee_id, leave_type, start_date, end_date, days_taken, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
        leaves,
    )

    # 7. Seed Performance Reviews
    reviews = [
        (1, 7, 2025, 4.9, "Outstanding engineering impact and system architecture leadership.", "2025-12-01"),
        (2, 1, 2025, 4.8, "Exceptional department leadership and technical strategy execution.", "2025-12-01"),
        (3, 3, 2025, 4.7, "Superb talent acquisition transformation and organizational design.", "2025-12-01"),
        (4, 18, 2025, 4.6, "Very strong business partnership and performance management.", "2025-12-01"),
        (5, 13, 2025, 4.6, "Exceeded revenue quota by 130%.", "2025-12-01"),
    ]
    conn.executemany(
        "INSERT INTO performance_reviews (id, employee_id, review_year, rating, review_comments, review_date) VALUES (?, ?, ?, ?, ?, ?)",
        reviews,
    )

    conn.commit()
    conn.close()
    print(f"Database successfully seeded at: {db_path}")



if __name__ == "__main__":
    seed_database()
