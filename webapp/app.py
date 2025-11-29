from pathlib import Path
import os
from functools import wraps

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    jsonify,
)
from werkzeug.security import generate_password_hash, check_password_hash
import psycopg
from psycopg import errors
from psycopg.rows import dict_row

from predictor import Predictor
from mailer import Mailer


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "best_gbv_model"
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:Wamaya2003@localhost:5432/femicide_db"
)
VIOLENCE_CATEGORIES = [
    "Physical_violence",
    "sexual_violence",
    "emotional_violence",
    "economic_violence",
]
FOCUS_AREAS = ["Nairobi", "Kisumu", "Mombasa"]

HELPLINE_FOOTER = """

Your case has been recorded. If you experience any further danger or need immediate support, please contact these national helplines:

• 1195 – Kenya's national GBV hotline
• 1190 – Counselling hotline
• 1517 – UNHCR toll-free help line
• 0800-720600 – Telecounselling AMANI Counselling Center
• 0800-720-050 – Legal services hotline
"""


def get_db():
    conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
    return conn


def init_db():
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS admins (
                    id SERIAL PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS reporters (
                    id SERIAL PRIMARY KEY,
                    full_name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    phone TEXT,
                    password_hash TEXT NOT NULL,
                    active BOOLEAN NOT NULL DEFAULT TRUE,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS reports (
                    id SERIAL PRIMARY KEY,
                    reporter_id INTEGER NOT NULL REFERENCES reporters(id) ON DELETE CASCADE,
                    complainant_name TEXT,
                    complainant_email TEXT,
                    complainant_phone TEXT,
                    address TEXT,
                    area TEXT,
                    subject TEXT,
                    message TEXT NOT NULL,
                    predicted_label TEXT NOT NULL,
                    confidence DOUBLE PRECISION NOT NULL,
                    verified BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS counselors (
                    id SERIAL PRIMARY KEY,
                    full_name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    phone TEXT,
                    category TEXT NOT NULL,
                    active BOOLEAN NOT NULL DEFAULT TRUE,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cur.execute(
                """
                ALTER TABLE reports
                ADD COLUMN IF NOT EXISTS counselor_id INTEGER REFERENCES counselors(id)
                """
            )
        conn.commit()


def has_admins() -> bool:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS total FROM admins")
            total = cur.fetchone()["total"]
    return total > 0


def create_admin(username: str, password_hash: str):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO admins (username, password_hash) VALUES (%s, %s)",
                (username.strip(), password_hash),
            )
        conn.commit()


def get_admin_by_username(username: str):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, username, password_hash
                FROM admins
                WHERE username = %s
                """,
                (username.strip(),),
            )
            return cur.fetchone()


def create_reporter(
    full_name: str,
    email: str,
    phone: str | None,
    password_hash: str,
    active: bool = True,
) -> None:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO reporters (full_name, email, phone, password_hash, active)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    full_name.strip(),
                    email.lower().strip(),
                    phone.strip() if phone else None,
                    password_hash,
                    active,
                ),
            )
        conn.commit()


def list_reporters():
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, full_name, email, phone, active, created_at
                FROM reporters
                ORDER BY active DESC, full_name ASC
                """
            )
            return cur.fetchall()


def set_reporter_active(reporter_id: int, active: bool):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE reporters SET active = %s WHERE id = %s",
                (active, reporter_id),
            )
        conn.commit()


def get_reporter_by_email(email: str):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, full_name, email, phone, password_hash, active
                FROM reporters
                WHERE email = %s
                """,
                (email.lower().strip(),),
            )
            return cur.fetchone()


def get_reporter_by_id(reporter_id: int):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, full_name, email, phone, active
                FROM reporters
                WHERE id = %s
                """,
                (reporter_id,),
            )
            return cur.fetchone()


def save_report(
    reporter_id: int,
    complainant_name: str | None,
    complainant_email: str | None,
    complainant_phone: str | None,
    address: str | None,
    area: str | None,
    subject: str | None,
    message: str,
    label: str,
    confidence: float,
    counselor_id: int | None = None,
):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO reports (
                    reporter_id,
                    complainant_name,
                    complainant_email,
                    complainant_phone,
                    address,
                    area,
                    subject,
                    message,
                    predicted_label,
                    confidence,
                    verified,
                    counselor_id
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, FALSE, %s)
                """,
                (
                    reporter_id,
                    complainant_name,
                    complainant_email,
                    complainant_phone,
                    address,
                    area,
                    subject,
                    message,
                    label,
                    confidence,
                    counselor_id,
                ),
            )
        conn.commit()


def fetch_reports_for_reporter(reporter_id: int):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT r.id,
                       r.subject,
                       r.complainant_name,
                       r.complainant_email,
                       r.complainant_phone,
                       r.address,
                       r.area,
                       r.message,
                       r.predicted_label,
                       r.confidence,
                       r.verified,
                       r.created_at,
                       c.full_name AS counselor_name,
                       c.email AS counselor_email,
                       c.phone AS counselor_phone
                FROM reports r
                LEFT JOIN counselors c ON c.id = r.counselor_id
                WHERE r.reporter_id = %s
                ORDER BY r.created_at DESC
                """,
                (reporter_id,),
            )
            return cur.fetchall()


def fetch_all_reports():
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT r.id,
                       r.subject,
                       r.complainant_name,
                       r.complainant_email,
                       r.complainant_phone,
                       r.address,
                       r.area,
                       r.message,
                       r.predicted_label,
                       r.confidence,
                       r.verified,
                       r.created_at,
                       rep.full_name AS reporter_name,
                       rep.email AS reporter_email,
                       c.full_name AS counselor_name,
                       c.email AS counselor_email,
                       c.phone AS counselor_phone
                FROM reports r
                LEFT JOIN reporters rep ON rep.id = r.reporter_id
                LEFT JOIN counselors c ON c.id = r.counselor_id
                ORDER BY r.created_at DESC
                """
            )
            return cur.fetchall()


def get_category_summary():
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT predicted_label, COUNT(*) AS total
                FROM reports
                GROUP BY predicted_label
                """
            )
            return cur.fetchall()


def get_report_metrics():
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS total FROM reports")
            total = cur.fetchone()["total"]
            cur.execute("SELECT COUNT(*) AS verified FROM reports WHERE verified = TRUE")
            verified = cur.fetchone()["verified"]
            cur.execute(
                """
                SELECT COUNT(*) AS recent
                FROM reports
                WHERE created_at >= (CURRENT_TIMESTAMP - INTERVAL '30 days')
                """
            )
            last_30 = cur.fetchone()["recent"]
    return {"total": total, "verified": verified, "pending": total - verified, "last_30": last_30}


def get_daily_category_counts():
    query = """
        SELECT DATE(created_at) AS day,
               predicted_label,
               COUNT(*) AS total
        FROM reports
        WHERE created_at >= (CURRENT_TIMESTAMP - INTERVAL '7 days')
        GROUP BY day, predicted_label
        ORDER BY day ASC
    """
    data = {}
    max_total = 0
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            for row in cur.fetchall():
                key = row["day"]
                data.setdefault(key, {cat: 0 for cat in VIOLENCE_CATEGORIES})
                data[key][row["predicted_label"]] = row["total"]
    for counts in data.values():
        total = sum(counts.values())
        if total > max_total:
            max_total = total
    return data, max_total


def get_area_case_counts():
    query = """
        SELECT
            CASE
                WHEN lower(area) LIKE 'nairobi%%' THEN 'Nairobi'
                WHEN lower(area) LIKE 'kisumu%%' THEN 'Kisumu'
                WHEN lower(area) LIKE 'mombasa%%' THEN 'Mombasa'
                ELSE 'Other'
            END AS region,
            COUNT(*) AS total
        FROM reports
        GROUP BY region
    """
    counts = {area: 0 for area in FOCUS_AREAS}
    counts["Other"] = 0
    max_count = 0
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            for row in cur.fetchall():
                counts[row["region"]] = row["total"]
                if row["total"] > max_count:
                    max_count = row["total"]
    return counts, max_count

def create_counselor(full_name: str, email: str, phone: str | None, category: str, active: bool = True):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO counselors (full_name, email, phone, category, active)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    full_name.strip(),
                    email.lower().strip(),
                    phone.strip() if phone else None,
                    category,
                    active,
                ),
            )
        conn.commit()


def list_counselors():
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, full_name, email, phone, category, active, created_at
                FROM counselors
                ORDER BY active DESC, category ASC, full_name ASC
                """
            )
            return cur.fetchall()


def set_counselor_active(counselor_id: int, active: bool):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE counselors SET active = %s WHERE id = %s",
                (active, counselor_id),
            )
        conn.commit()


def get_counselor_for_category(category: str):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, full_name, email, phone
                FROM counselors
                WHERE active = TRUE AND category = %s
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (category,),
            )
            return cur.fetchone()


def get_report_for_reporter(report_id: int, reporter_id: int):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT r.id,
                       r.reporter_id,
                       r.complainant_name,
                       r.complainant_email,
                       r.complainant_phone,
                       r.subject,
                       r.message,
                       r.predicted_label,
                       r.verified,
                       r.created_at,
                       c.full_name AS counselor_name,
                       c.email AS counselor_email,
                       c.phone AS counselor_phone
                FROM reports r
                LEFT JOIN counselors c ON c.id = r.counselor_id
                WHERE r.id = %s AND r.reporter_id = %s
                """,
                (report_id, reporter_id),
            )
            return cur.fetchone()


def get_report_by_id(report_id: int):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT r.*,
                       c.full_name AS counselor_name,
                       c.email AS counselor_email,
                       c.phone AS counselor_phone
                FROM reports r
                LEFT JOIN counselors c ON c.id = r.counselor_id
                WHERE r.id = %s
                """,
                (report_id,),
            )
            return cur.fetchone()


def reporter_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("reporter_id"):
            flash("Please sign in to continue.", "warning")
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped


def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("admin_id"):
            flash("Admin login required.", "warning")
            return redirect(url_for("admin_login"))
        return view(*args, **kwargs)

    return wrapped


def create_app():
    app = Flask(__name__, template_folder="templates")
    app.config["SECRET_KEY"] = "change-me-in-prod"
    mail_server = os.environ.get("MAIL_SERVER", "smtp.gmail.com")
    mail_port = int(os.environ.get("MAIL_PORT", "465"))
    mail_username = os.environ.get("MAIL_USERNAME", "wamayateddy@gmail.com")
    mail_password = os.environ.get("MAIL_PASSWORD", "gtpq gaui hbmy epxz")
    mail_sender = os.environ.get("MAIL_SENDER", f"Femicide Watch <{mail_username}>")
    mail_use_ssl = os.environ.get("MAIL_USE_SSL", "true").lower() != "false"
    low_conf_threshold = float(os.environ.get("LOW_CONFIDENCE_THRESHOLD", "0.7"))
    app.config.update(
        MAIL_SERVER=mail_server,
        MAIL_PORT=mail_port,
        MAIL_USERNAME=mail_username,
        MAIL_PASSWORD=mail_password,
        MAIL_SENDER=mail_sender,
        MAIL_USE_SSL=mail_use_ssl,
        LOW_CONFIDENCE_THRESHOLD=low_conf_threshold,
    )

    init_db()
    predictor = Predictor(MODEL_DIR)
    mailer = Mailer(
        server=app.config["MAIL_SERVER"],
        port=app.config["MAIL_PORT"],
        username=app.config["MAIL_USERNAME"],
        password=app.config["MAIL_PASSWORD"],
        sender=app.config["MAIL_SENDER"],
        use_ssl=app.config["MAIL_USE_SSL"],
    )
    try:
        predictor.predict("warmup request")
    except Exception:
        pass

    def build_prediction_payload(label, confidence, probabilities, **extra):
        ordered_probs = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        payload = {
            "label": label,
            "confidence_value": confidence,
            "confidence": f"{confidence:.2%}",
            "probabilities": ordered_probs,
            "low_confidence": confidence < app.config["LOW_CONFIDENCE_THRESHOLD"],
        }
        payload.update(extra)
        return payload

    @app.context_processor
    def inject_globals():
        return {
            "session": session,
        }

    @app.after_request
    def add_headers(response):
        response.headers.setdefault("Cache-Control", "no-store")
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        return response

    @app.route("/")
    def landing():
        return render_template("landing.html")

    @app.route("/about")
    def about():
        return render_template("about.html")

    @app.route("/contact")
    def contact():
        return render_template("contact.html")

    @app.route("/demo-report", methods=["GET", "POST"])
    def demo_report():
        age_ranges = [
            ("under_18", "Under 18"),
            ("18_24", "18 - 24"),
            ("25_34", "25 - 34"),
            ("35_44", "35 - 44"),
            ("45_54", "45 - 54"),
            ("55_plus", "55 and above"),
        ]
        valid_age_ranges = {value for value, _ in age_ranges}
        prediction = None
        form_values = {
            "full_name": request.form.get("full_name", "").strip(),
            "phone": request.form.get("phone", "").strip(),
            "age_range": request.form.get("age_range", ""),
            "message": request.form.get("message", "").strip(),
        }
        if request.method == "POST":
            errors = []
            if not form_values["full_name"]:
                errors.append("Please provide your name.")
            if not form_values["phone"]:
                errors.append("Phone number is required.")
            if form_values["age_range"] not in valid_age_ranges:
                errors.append("Select an age range.")
            if len(form_values["message"]) < 15:
                errors.append("Report description should be at least 15 characters.")
            if errors:
                for err in errors:
                    flash(err, "warning")
            else:
                try:
                    label, confidence, probabilities = predictor.predict(form_values["message"])
                    counselor = get_counselor_for_category(label)
                    prediction = build_prediction_payload(
                        label,
                        confidence,
                        probabilities,
                        full_name=form_values["full_name"],
                        age_range=form_values["age_range"],
                        phone=form_values["phone"],
                        message=form_values["message"],
                        counselor=counselor,
                    )
                except Exception as exc:
                    flash(f"Prediction error: {exc}", "danger")
        return render_template(
            "demo_report.html",
            age_ranges=age_ranges,
            form_values=form_values,
            prediction=prediction,
        )

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            reporter = get_reporter_by_email(email)
            if not reporter:
                flash("Account not found.", "danger")
            elif not reporter["active"]:
                flash("This account is inactive. Contact the admin.", "danger")
            elif check_password_hash(reporter["password_hash"], password):
                session.clear()
                session["reporter_id"] = reporter["id"]
                session["reporter_name"] = reporter["full_name"]
                session["reporter_email"] = reporter["email"]
                flash("Welcome back!", "success")
                return redirect(url_for("dashboard"))
            else:
                flash("Invalid credentials.", "danger")
        return render_template("login.html")

    @app.route("/logout")
    def logout():
        session.clear()
        flash("Signed out successfully.", "info")
        return redirect(url_for("landing"))

    @app.route("/dashboard", methods=["GET", "POST"])
    @reporter_required
    def dashboard():
        reporter = get_reporter_by_id(session["reporter_id"])
        prediction = None
        if request.method == "POST":
            subject = request.form.get("subject", "").strip()
            complainant_name = request.form.get("complainant_name", "").strip()
            complainant_email = request.form.get("complainant_email", "").strip()
            complainant_phone = request.form.get("complainant_phone", "").strip()
            address = request.form.get("address", "").strip()
            area = request.form.get("area", "").strip()
            message = request.form.get("message", "").strip()
            if not message:
                flash("Incident description is required.", "warning")
            else:
                try:
                    label, confidence, probabilities = predictor.predict(message)
                    counselor = get_counselor_for_category(label)
                    save_report(
                        reporter_id=reporter["id"],
                        complainant_name=complainant_name or None,
                        complainant_email=complainant_email or None,
                        complainant_phone=complainant_phone or None,
                        address=address or None,
                        area=area or None,
                        subject=subject or None,
                        message=message,
                        label=label,
                        confidence=confidence,
                        counselor_id=counselor["id"] if counselor else None,
                    )
                    prediction = build_prediction_payload(
                        label,
                        confidence,
                        probabilities,
                        subject=subject,
                        message=message,
                        counselor=counselor,
                    )
                    flash("Report filed successfully.", "success")
                except Exception as exc:
                    flash(f"Prediction error: {exc}", "danger")
        reports = fetch_reports_for_reporter(reporter["id"])
        return render_template("dashboard.html", reporter=reporter, reports=reports, prediction=prediction)

    @app.route("/reports/<int:report_id>/email", methods=["GET", "POST"])
    @reporter_required
    def email_complainant(report_id: int):
        report = get_report_for_reporter(report_id, session["reporter_id"])
        if not report:
            flash("Report not found.", "danger")
            return redirect(url_for("dashboard"))
        if not report["complainant_email"]:
            flash("Complaints without an email cannot be contacted through this form.", "warning")
            return redirect(url_for("dashboard"))

        default_subject = f"Follow-up on your report ({report['subject'] or report['predicted_label']})"
        subject = request.form.get("subject", default_subject).strip() or default_subject
        message = request.form.get("message", "").strip()

        if request.method == "POST":
            if len(message) < 10:
                flash("Message should be at least 10 characters.", "warning")
            else:
                try:
                    message_body = f"{message.strip()}\n{HELPLINE_FOOTER}"
                    mailer.send(report["complainant_email"], subject, message_body)
                    flash("Email sent successfully.", "success")
                    return redirect(url_for("dashboard"))
                except Exception as exc:
                    flash(f"Email failed: {exc}", "danger")
        return render_template("email_complainant.html", report=report, subject=subject, message=message)

    @app.route("/admin/register", methods=["GET", "POST"])
    def admin_register():
        bootstrap_mode = not has_admins()
        if not bootstrap_mode and not session.get("admin_id"):
            flash("Only logged-in admins can add more admins.", "warning")
            return redirect(url_for("admin_login"))
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")
            confirm = request.form.get("confirm_password", "")
            if not username or not password:
                flash("Username and password are required.", "danger")
            elif password != confirm:
                flash("Passwords do not match.", "danger")
            else:
                try:
                    create_admin(username, generate_password_hash(password))
                    flash("Admin account created.", "success")
                    if bootstrap_mode:
                        return redirect(url_for("admin_login"))
                except errors.UniqueViolation:
                    flash("Username already exists.", "danger")
        return render_template("admin_register.html", bootstrap=bootstrap_mode)

    @app.route("/admin/login", methods=["GET", "POST"])
    def admin_login():
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")
            admin = get_admin_by_username(username)
            if admin and check_password_hash(admin["password_hash"], password):
                session.clear()
                session["admin_id"] = admin["id"]
                session["admin_username"] = admin["username"]
                flash("Admin login successful.", "success")
                return redirect(url_for("admin_reporters"))
            flash("Invalid admin credentials.", "danger")
        return render_template("admin_login.html")

    @app.route("/admin/logout")
    def admin_logout():
        session.clear()
        flash("Admin logged out.", "info")
        return redirect(url_for("landing"))

    @app.route("/admin/reporters", methods=["GET", "POST"])
    @admin_required
    def admin_reporters():
        if request.method == "POST":
            full_name = request.form.get("full_name", "").strip()
            email = request.form.get("email", "").strip().lower()
            phone = request.form.get("phone", "").strip()
            password = request.form.get("password", "")
            active = bool(request.form.get("active"))
            if not full_name or not email or not password:
                flash("Name, email, and password are required.", "warning")
            else:
                try:
                    create_reporter(
                        full_name=full_name,
                        email=email,
                        phone=phone or None,
                        password_hash=generate_password_hash(password),
                        active=active,
                    )
                    flash("Reporter added.", "success")
                except errors.UniqueViolation:
                    flash("A reporter with that email already exists.", "danger")
        reporters = list_reporters()
        return render_template("admin_reporters.html", reporters=reporters)

    @app.route("/admin/reporters/<int:reporter_id>/reports")
    @admin_required
    def admin_reporter_reports(reporter_id: int):
        reporter = get_reporter_by_id(reporter_id)
        if not reporter:
            flash("Reporter not found.", "danger")
            return redirect(url_for("admin_reporters"))
        reports = fetch_reports_for_reporter(reporter_id)
        return render_template("admin_reporter_reports.html", reporter=reporter, reports=reports)

    @app.route("/admin/reporters/<int:reporter_id>/reports/<int:report_id>")
    @admin_required
    def admin_reporter_report_detail(reporter_id: int, report_id: int):
        reporter = get_reporter_by_id(reporter_id)
        if not reporter:
            flash("Reporter not found.", "danger")
            return redirect(url_for("admin_reporters"))
        report = get_report_by_id(report_id)
        if not report or report["reporter_id"] != reporter_id:
            flash("Report not found.", "warning")
            return redirect(url_for("admin_reporter_reports", reporter_id=reporter_id))
        return render_template("admin_report_detail.html", reporter=reporter, report=report)

    @app.route("/admin/counselors", methods=["GET", "POST"])
    @admin_required
    def admin_counselors():
        if request.method == "POST":
            full_name = request.form.get("full_name", "").strip()
            email = request.form.get("email", "").strip()
            phone = request.form.get("phone", "").strip()
            category = request.form.get("category", "")
            if not full_name or not email or category not in VIOLENCE_CATEGORIES:
                flash("Name, valid category, and email are required.", "warning")
            else:
                try:
                    create_counselor(
                        full_name=full_name,
                        email=email,
                        phone=phone or None,
                        category=category,
                        active=True,
                    )
                    flash("Counselor added.", "success")
                except errors.UniqueViolation:
                    flash("A counselor with that email already exists.", "danger")
        counselors = list_counselors()
        return render_template("admin_counselors.html", counselors=counselors, categories=VIOLENCE_CATEGORIES)

    @app.route("/admin/counselors/<int:counselor_id>/toggle", methods=["POST"])
    @admin_required
    def toggle_counselor(counselor_id: int):
        action = request.form.get("action")
        set_counselor_active(counselor_id, action != "deactivate")
        flash("Counselor status updated.", "info")
        return redirect(url_for("admin_counselors"))

    @app.route("/admin/reporters/<int:reporter_id>/toggle", methods=["POST"])
    @admin_required
    def toggle_reporter_status(reporter_id: int):
        action = request.form.get("action")
        if action == "deactivate":
            set_reporter_active(reporter_id, False)
            flash("Reporter deactivated.", "info")
        else:
            set_reporter_active(reporter_id, True)
            flash("Reporter activated.", "success")
        return redirect(url_for("admin_reporters"))

    @app.route("/admin/reports")
    @admin_required
    def admin_reports():
        reports = fetch_all_reports()
        stats = get_category_summary()
        return render_template("admin_reports.html", reports=reports, stats=stats)

    @app.route("/admin/metrics")
    @admin_required
    def admin_metrics():
        stats = get_category_summary()
        metrics = get_report_metrics()
        reporters = list_reporters()
        counselors = list_counselors()
        daily, max_daily = get_daily_category_counts()
        area_counts, max_area = get_area_case_counts()
        return render_template(
            "admin_metrics.html",
            stats=stats,
            metrics=metrics,
            reporters=reporters,
            counselors=counselors,
            daily=daily,
            max_daily=max_daily,
            categories=VIOLENCE_CATEGORIES,
            area_counts=area_counts,
            max_area=max_area,
        )

    @app.route("/admin/reports/<int:report_id>/verify", methods=["POST"])
    @admin_required
    def admin_verify_report(report_id: int):
        verified = bool(request.form.get("verified"))
        redirect_to = request.form.get("redirect_to") or url_for("admin_reports")
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE reports SET verified = %s WHERE id = %s",
                    (verified, report_id),
                )
            conn.commit()
        flash("Report verification updated.", "success")
        return redirect(redirect_to)

    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        data = request.get_json(silent=True) or {}
        message = data.get("message") or data.get("text")
        if not message:
            return jsonify({"ok": False, "error": "Missing message"}), 400
        try:
            label, confidence, probabilities = predictor.predict(message)
            return (
                jsonify(
                    {
                        "ok": True,
                        "label": label,
                        "confidence": confidence,
                        "probabilities": probabilities,
                        "low_confidence": confidence < app.config["LOW_CONFIDENCE_THRESHOLD"],
                    }
                ),
                200,
            )
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

