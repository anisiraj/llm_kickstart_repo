-- ============================================================================
-- PLACEHOLDER: Company Database Schema
-- ============================================================================
-- This represents a typical mid-size org's operational database.
-- Replace with YOUR org's actual schema (anonymized) for the bootcamp exercise.
--
-- Tables: employees, departments, projects, timesheets, incidents, customers,
--         contracts, invoices, equipment, maintenance_logs
-- ============================================================================

-- ── Core HR ─────────────────────────────────────────────────────────────────

CREATE TABLE departments (
    dept_id         SERIAL PRIMARY KEY,
    dept_name       VARCHAR(100) NOT NULL,
    cost_center     VARCHAR(20),
    parent_dept_id  INTEGER REFERENCES departments(dept_id),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE employees (
    emp_id          SERIAL PRIMARY KEY,
    first_name      VARCHAR(50) NOT NULL,
    last_name       VARCHAR(50) NOT NULL,
    email           VARCHAR(120) UNIQUE NOT NULL,
    dept_id         INTEGER REFERENCES departments(dept_id),
    manager_id      INTEGER REFERENCES employees(emp_id),
    hire_date       DATE NOT NULL,
    role_title      VARCHAR(100),
    salary_band     VARCHAR(10),  -- 'L1'..'L8'
    location        VARCHAR(80),
    is_active       BOOLEAN DEFAULT TRUE
);

CREATE TABLE timesheets (
    ts_id           SERIAL PRIMARY KEY,
    emp_id          INTEGER REFERENCES employees(emp_id),
    project_id      INTEGER REFERENCES projects(project_id),
    work_date       DATE NOT NULL,
    hours           NUMERIC(4,2) NOT NULL CHECK (hours > 0 AND hours <= 24),
    category        VARCHAR(30),  -- 'engineering', 'meeting', 'oncall', 'admin'
    notes           TEXT
);

-- ── Projects & Delivery ─────────────────────────────────────────────────────

CREATE TABLE projects (
    project_id      SERIAL PRIMARY KEY,
    project_name    VARCHAR(150) NOT NULL,
    dept_id         INTEGER REFERENCES departments(dept_id),
    lead_emp_id     INTEGER REFERENCES employees(emp_id),
    status          VARCHAR(20) DEFAULT 'active',  -- 'active','paused','completed','cancelled'
    start_date      DATE,
    target_end_date DATE,
    actual_end_date DATE,
    budget          NUMERIC(12,2),
    priority        VARCHAR(10)  -- 'P0','P1','P2','P3'
);

CREATE TABLE project_members (
    project_id      INTEGER REFERENCES projects(project_id),
    emp_id          INTEGER REFERENCES employees(emp_id),
    role            VARCHAR(50),  -- 'lead','engineer','analyst','reviewer'
    joined_date     DATE,
    PRIMARY KEY (project_id, emp_id)
);

-- ── Customers & Revenue ─────────────────────────────────────────────────────

CREATE TABLE customers (
    customer_id     SERIAL PRIMARY KEY,
    company_name    VARCHAR(200) NOT NULL,
    industry        VARCHAR(80),
    region          VARCHAR(50),   -- 'NA','EMEA','APAC','LATAM'
    tier            VARCHAR(10),   -- 'enterprise','mid-market','smb'
    acq_date        DATE,
    csm_emp_id      INTEGER REFERENCES employees(emp_id),
    arr             NUMERIC(12,2), -- annual recurring revenue
    is_churned      BOOLEAN DEFAULT FALSE
);

CREATE TABLE contracts (
    contract_id     SERIAL PRIMARY KEY,
    customer_id     INTEGER REFERENCES customers(customer_id),
    contract_type   VARCHAR(30),   -- 'subscription','professional_services','support'
    start_date      DATE NOT NULL,
    end_date        DATE NOT NULL,
    value           NUMERIC(12,2),
    auto_renew      BOOLEAN DEFAULT TRUE,
    signed_by_emp   INTEGER REFERENCES employees(emp_id)
);

CREATE TABLE invoices (
    invoice_id      SERIAL PRIMARY KEY,
    contract_id     INTEGER REFERENCES contracts(contract_id),
    customer_id     INTEGER REFERENCES customers(customer_id),
    issue_date      DATE NOT NULL,
    due_date        DATE NOT NULL,
    amount          NUMERIC(12,2) NOT NULL,
    status          VARCHAR(20) DEFAULT 'pending',  -- 'pending','paid','overdue','void'
    paid_date       DATE
);

-- ── Operations / Incidents ──────────────────────────────────────────────────

CREATE TABLE equipment (
    equip_id        SERIAL PRIMARY KEY,
    equip_name      VARCHAR(150) NOT NULL,
    equip_type      VARCHAR(50),   -- 'hvac','chiller','rooftop_unit','ahu','vrf'
    site_id         VARCHAR(30),
    customer_id     INTEGER REFERENCES customers(customer_id),
    install_date    DATE,
    warranty_end    DATE,
    model_number    VARCHAR(80),
    serial_number   VARCHAR(80) UNIQUE
);

CREATE TABLE maintenance_logs (
    log_id          SERIAL PRIMARY KEY,
    equip_id        INTEGER REFERENCES equipment(equip_id),
    technician_id   INTEGER REFERENCES employees(emp_id),
    log_date        DATE NOT NULL,
    work_type       VARCHAR(30),   -- 'preventive','corrective','emergency'
    description     TEXT,
    parts_used      TEXT,
    duration_hours  NUMERIC(4,2),
    cost            NUMERIC(10,2)
);

CREATE TABLE incidents (
    incident_id     SERIAL PRIMARY KEY,
    customer_id     INTEGER REFERENCES customers(customer_id),
    equip_id        INTEGER REFERENCES equipment(equip_id),
    reported_by     INTEGER REFERENCES employees(emp_id),
    assigned_to     INTEGER REFERENCES employees(emp_id),
    severity        VARCHAR(10),   -- 'SEV1','SEV2','SEV3','SEV4'
    status          VARCHAR(20) DEFAULT 'open', -- 'open','investigating','resolved','closed'
    opened_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at     TIMESTAMP,
    root_cause      TEXT,
    resolution      TEXT
);

-- ── Indexes (representative, not exhaustive) ────────────────────────────────

CREATE INDEX idx_emp_dept       ON employees(dept_id);
CREATE INDEX idx_emp_manager    ON employees(manager_id);
CREATE INDEX idx_ts_emp_date    ON timesheets(emp_id, work_date);
CREATE INDEX idx_proj_status    ON projects(status);
CREATE INDEX idx_cust_region    ON customers(region);
CREATE INDEX idx_inv_status     ON invoices(status);
CREATE INDEX idx_inc_severity   ON incidents(severity, status);
CREATE INDEX idx_maint_equip    ON maintenance_logs(equip_id, log_date);
