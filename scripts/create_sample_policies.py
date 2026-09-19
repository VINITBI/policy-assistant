"""Generate 4 realistic demo policy PDFs in data/sample_policies/ (to try the system before using real policies).

    python scripts/create_sample_policies.py
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path(__file__).resolve().parents[1] / "data" / "sample_policies"

POLICIES = {
    "Leave_Policy.pdf": ("Leave Policy", [
        ("1. Purpose", "This policy defines the types of leave available to employees of the Company and the rules "
                       "for requesting and approving leave. It applies to all full-time and part-time employees."),
        ("2. Annual Leave", "Full-time employees accrue 1.5 days of annual leave per completed month of service, "
                            "i.e. 18 days per calendar year. Part-time employees accrue leave on a pro-rata basis. "
                            "Annual leave must be requested at least 7 calendar days in advance through the HR portal "
                            "and approved by the reporting manager."),
        ("2.1 Carry Forward", "A maximum of 10 unused annual leave days may be carried forward to the next calendar "
                              "year. Carried-forward leave must be used by 31 March, after which it lapses. Leave "
                              "cannot be encashed except at separation."),
        ("3. Sick Leave", "Employees are entitled to 12 days of paid sick leave per calendar year. Sick leave of more "
                          "than 2 consecutive days requires a medical certificate from a registered medical "
                          "practitioner, submitted within 3 working days of returning to work."),
        ("4. Maternity and Paternity Leave", "Female employees with at least 80 days of service in the preceding 12 "
                                             "months are entitled to 26 weeks of paid maternity leave. Male employees "
                                             "are entitled to 10 working days of paid paternity leave, to be taken "
                                             "within 3 months of the child's birth or adoption."),
        ("5. Leave Without Pay", "Leave without pay (LWP) may be granted for up to 30 days in exceptional "
                                 "circumstances once all paid leave is exhausted, subject to approval by the "
                                 "department head and HR."),
        ("6. Public Holidays", "The Company observes 10 public holidays per year, published by HR in December for the "
                               "following year. Employees required to work on a public holiday receive a "
                               "compensatory day off within 30 days."),
    ], [["Leave type", "Entitlement", "Documentation"], ["Annual", "18 days / year", "HR portal request"],
         ["Sick", "12 days / year", "Certificate if > 2 days"], ["Maternity", "26 weeks", "Medical proof"],
         ["Paternity", "10 working days", "Birth / adoption record"]]),
    "Travel_and_Expense_Policy.pdf": ("Travel and Expense Policy", [
        ("1. Scope", "This policy governs business travel and reimbursement of business expenses for all employees."),
        ("2. Travel Approval", "All business travel must be pre-approved by the reporting manager. International "
                               "travel additionally requires approval from the department head at least 14 days "
                               "before departure."),
        ("3. Air Travel", "Economy class is mandatory for flights under 6 hours. Business class may be booked for "
                          "flights of 6 hours or more for employees at Director level and above."),
        ("4. Accommodation", "Hotel stays are reimbursed up to INR 6,000 per night in metro cities and INR 4,000 per "
                             "night in other cities. International hotel limits are USD 180 per night."),
        ("5. Meals and Per Diem", "A daily meal allowance of INR 1,500 applies for domestic travel and USD 60 for "
                                  "international travel. Alcohol is not reimbursable."),
        ("6. Expense Claims", "Expense claims must be submitted with original receipts within 30 days of the trip "
                              "end date via the expense portal. Claims submitted after 60 days will not be "
                              "reimbursed. Approved claims are paid with the next monthly payroll."),
    ], [["Item", "Domestic limit", "International limit"], ["Hotel (metro)", "INR 6,000/night", "USD 180/night"],
         ["Hotel (other)", "INR 4,000/night", "USD 180/night"], ["Meals", "INR 1,500/day", "USD 60/day"]]),
    "Work_From_Home_Policy.pdf": ("Work From Home Policy", [
        ("1. Eligibility", "Employees who have completed their probation period of 6 months are eligible for work "
                           "from home (WFH), subject to role suitability and manager approval."),
        ("2. WFH Allowance", "Eligible employees may work from home up to 2 days per week. Fully remote arrangements "
                             "require approval from the department head and HR."),
        ("3. Equipment", "The Company provides a laptop and a one-time home office allowance of INR 10,000 for "
                         "furniture or peripherals, claimable within 90 days of WFH approval."),
        ("4. Working Hours", "Employees working from home must be available during core hours of 11:00 to 16:00 "
                             "local time and must respond to messages within 30 minutes during core hours."),
        ("5. Data Security", "Employees must connect through the corporate VPN, must not use public Wi-Fi without VPN, "
                             "and must lock their screen when away. Printing confidential documents at home is "
                             "prohibited."),
    ], None),
    "IT_Acceptable_Use_Policy.pdf": ("IT Acceptable Use Policy", [
        ("1. Purpose", "This policy sets rules for the acceptable use of Company IT systems, devices and data."),
        ("2. Passwords", "Passwords must be at least 12 characters and include upper-case, lower-case, numeric and "
                         "special characters. Passwords must be changed every 90 days and must never be shared. "
                         "Multi-factor authentication is mandatory for email and VPN."),
        ("3. Software", "Only software approved by the IT department may be installed on Company devices. Requests "
                        "for new software are raised through the IT service desk and are reviewed within 5 working days."),
        ("4. Personal Devices (BYOD)", "Personal devices may access Company email only after enrolment in the mobile "
                                       "device management (MDM) system. Company data must not be stored on personal "
                                       "cloud storage."),
        ("5. Incident Reporting", "Suspected security incidents, phishing emails or lost devices must be reported to "
                                  "the IT security team at security@company.example within 1 hour of discovery."),
        ("6. Violations", "Violations of this policy may result in disciplinary action up to and including "
                          "termination of employment."),
    ], None),
}


def build(file_name: str, title: str, sections: list[tuple[str, str]], table: list[list[str]] | None) -> None:
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(OUT / file_name), pagesize=A4, title=title)
    story = [Paragraph(title, styles["Title"]), Paragraph("Version 1.0 - Effective 1 April 2026", styles["Normal"]),
             Spacer(1, 12)]
    for i, (heading, body) in enumerate(sections):
        story += [Paragraph(heading, styles["Heading2"]), Paragraph(body, styles["BodyText"]), Spacer(1, 8)]
        if i == 3:
            story.append(PageBreak())
    if table:
        story += [Paragraph("Annexure A - Summary", styles["Heading2"]), Spacer(1, 6)]
        t = Table(table)
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, "grey"), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]))
        story.append(t)
    doc.build(story)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for fname, (title, sections, table) in POLICIES.items():
        build(fname, title, sections, table)
        print("created", OUT / fname)
