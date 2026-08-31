-- PeopleQuery AI: HR Database Schema (SQLite / PostgreSQL compatible)

CREATE TABLE IF NOT EXISTS departments (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    code VARCHAR(10) NOT NULL UNIQUE,
    location VARCHAR(100) NOT NULL,
    budget NUMERIC(12, 2) NOT NULL
);

CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY,
    title VARCHAR(100) NOT NULL,
    department_id INTEGER NOT NULL REFERENCES departments(id),
    min_salary NUMERIC(10, 2) NOT NULL,
    max_salary NUMERIC(10, 2) NOT NULL
);

CREATE TABLE IF NOT EXISTS employees (
    id INTEGER PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) NOT NULL UNIQUE,
    gender VARCHAR(10) NOT NULL,
    department_id INTEGER NOT NULL REFERENCES departments(id),
    position_id INTEGER NOT NULL REFERENCES positions(id),
    hire_date DATE NOT NULL,
    employment_status VARCHAR(20) NOT NULL DEFAULT 'ACTIVE', -- ACTIVE, ON_LEAVE, TERMINATED
    employment_type VARCHAR(20) NOT NULL DEFAULT 'FULL_TIME', -- FULL_TIME, PART_TIME, CONTRACTOR
    salary NUMERIC(10, 2) NOT NULL,
    performance_rating NUMERIC(3, 2) DEFAULT 3.00,
    manager_id INTEGER REFERENCES employees(id)
);

CREATE TABLE IF NOT EXISTS leaves (
    id INTEGER PRIMARY KEY,
    employee_id INTEGER NOT NULL REFERENCES employees(id),
    leave_type VARCHAR(30) NOT NULL, -- SICK, ANNUAL, MATERNITY, PATERNITY, UNPAID
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    days_taken INTEGER NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'APPROVED' -- APPROVED, PENDING, REJECTED
);

CREATE TABLE IF NOT EXISTS benefits (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    code VARCHAR(20) NOT NULL UNIQUE,
    description TEXT,
    min_tenure_months INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS employee_benefits (
    id INTEGER PRIMARY KEY,
    employee_id INTEGER NOT NULL REFERENCES employees(id),
    benefit_id INTEGER NOT NULL REFERENCES benefits(id),
    enrolled_date DATE NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'ACTIVE'
);

CREATE TABLE IF NOT EXISTS performance_reviews (
    id INTEGER PRIMARY KEY,
    employee_id INTEGER NOT NULL REFERENCES employees(id),
    review_year INTEGER NOT NULL,
    rating NUMERIC(3, 2) NOT NULL,
    review_comments TEXT,
    review_date DATE NOT NULL
);
