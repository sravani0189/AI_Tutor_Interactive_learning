import pandas as pd
from pathlib import Path

INPUT_FILE = "independent_learner_questions_to_write.csv"
OUTPUT_FILE = "independent_learner_questions_filled.csv"

QUESTIONS = {
    "IQ001": "What are some ways a company can protect itself against common computer security threats?",
    "IQ002": "How did the invention of the microprocessor change who could use computers?",
    "IQ003": "Why can cloud computing reduce the amount of storage a person needs on their own computer?",
    "IQ004": "Why would someone choose a spreadsheet application instead of a word processing application for numerical data?",
    "IQ005": "What happens to a file when you create it in a Google Workspace application?",
    "IQ006": "How can someone customize the Microsoft 365 ribbon and application view to better fit their needs?",
    "IQ007": "How can Track Changes and commenting help people review a Word document together?",
    "IQ008": "How does Suggesting mode help multiple people collaborate on the same Google Docs document?",
    "IQ009": "Why would you use a section break when different parts of a Google Docs document need different formatting?",
    "IQ010": "Which Word features can help a reader move through a long document more easily?",
    "IQ011": "What feature in Google Docs can you use when you want to create a checklist?",
    "IQ012": "What kinds of settings can be changed through the Word Options dialog box?",
    "IQ013": "Why do you need to enable the Developer tab when creating a fillable form in Word?",
    "IQ014": "How does Word combine a main document with recipient information when creating personalized letters?",
    "IQ015": "How can sharing permissions control what different people are allowed to do in a Google Docs document?",
    "IQ016": "Which Google Slides tools can help line up text boxes and images neatly on a slide?",
    "IQ017": "Why is it usually better to keep the amount of text on a PowerPoint slide limited?",
    "IQ018": "Why is arranging text graphics and other objects important when designing a PowerPoint slide?",
    "IQ019": "What should a presenter think about before giving a presentation to an audience?",
    "IQ020": "How can transitions help both the presenter and the audience during a Google Slides presentation?",
    "IQ021": "What PowerPoint features can help someone prepare their slides and practice before presenting?",
    "IQ022": "How can a business use social media to develop relationships with both existing and potential customers?",
    "IQ023": "What changes can a business make to a website to improve its chances of appearing higher in search results?",
    "IQ024": "How can a content management system help an organization maintain consistent content across different platforms?",
    "IQ025": "Which formatting features do Google Sheets and Excel have in common?",
    "IQ026": "What is the relationship between an Excel workbook and its worksheets?",
    "IQ027": "How does Google Sheets decide what type of graph to create after you select data?",
    "IQ028": "How can a PivotTable help a business reorganize data to reveal different insights?",
    "IQ029": "Why might a business use logical functions when analyzing its internal data?",
    "IQ030": "How can a company keep the formatting of its Excel workbooks consistent with its branding?",
    "IQ031": "When would you use Excel's Analysis ToolPak instead of only basic built-in statistical functions?",
    "IQ032": "When is Goal Seek useful for changing an input value to reach a desired result?",
    "IQ034": "Why might a business export accounting data to Excel instead of doing all of its analysis in accounting software?",
    "IQ035": "How can time-value-of-money variables help someone evaluate a financial decision over time?",
    "IQ036": "How do financial accounting and managerial accounting serve different decision-making needs?",
    "IQ037": "What can an Access query do with information stored in one or more database tables?",
    "IQ038": "Why might someone use a form instead of working directly with the underlying database?",
    "IQ039": "How can an Access report make database information more useful for business decision making?",
    "IQ040": "How can a dashboard with navigation forms make an Access database easier to use in an organization?",
    "IQ041": "Why might an Access form combine information from two related tables?",
    "IQ042": "How can macros reduce the effort required for tasks that are performed repeatedly in Access?",
    "IQ043": "What is the difference between embedding information and linking information between Microsoft 365 files?",
    "IQ044": "What options are available for bringing Excel content into a Microsoft Word document?",
}

# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

if not Path(INPUT_FILE).exists():
    raise FileNotFoundError(
        f"Missing file: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

required = [
    "question_id",
    "top_chapter",
    "source_section",
    "source_text",
    "independent_learner_question",
    "question_author_notes",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise RuntimeError(
        "Missing columns: " + ", ".join(missing)
    )

# ------------------------------------------------------------
# REMOVE IQ033
# ------------------------------------------------------------

df = df[
    df["question_id"].astype(str).str.strip() != "IQ033"
].copy()

# ------------------------------------------------------------
# FILL QUESTIONS
# ------------------------------------------------------------

df["independent_learner_question"] = (
    df["question_id"]
    .astype(str)
    .str.strip()
    .map(QUESTIONS)
)

missing_questions = df[
    df["independent_learner_question"].isna()
]

if len(missing_questions) > 0:
    print("Missing question mappings:")
    print(
        missing_questions[
            ["question_id", "source_section"]
        ].to_string(index=False)
    )
    raise RuntimeError(
        "Some question IDs do not have a mapped independent question."
    )

# Keep author notes blank
df["question_author_notes"] = ""

# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

question_series = (
    df["independent_learner_question"]
    .fillna("")
    .astype(str)
    .str.strip()
)

if len(df) != 43:
    raise RuntimeError(
        f"Expected 43 rows after removing IQ033; found {len(df)}."
    )

if (question_series == "").any():
    raise RuntimeError(
        "Some independent questions are blank."
    )

if question_series.duplicated().any():
    duplicates = question_series[
        question_series.duplicated(keep=False)
    ]

    print("Duplicate questions:")
    print(duplicates.to_string(index=False))

    raise RuntimeError(
        "Duplicate independent questions found."
    )

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

# ------------------------------------------------------------
# REPORT
# ------------------------------------------------------------

print()
print("==============================")
print("INDEPENDENT QUESTIONS FILLED")
print("==============================")

print("Rows:", len(df))
print("Filled:", int((question_series != "").sum()))
print("Blank:", int((question_series == "").sum()))
print("Duplicates:", int(question_series.duplicated().sum()))

print()
print("Chapter distribution:")

print(
    df["top_chapter"]
    .value_counts()
    .sort_index()
    .to_string()
)

print()
print("Saved:")
print(OUTPUT_FILE)

print()
print("PASS: 43 independent learner questions created.")