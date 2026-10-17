
import joblib

# Load the trained Linear SVM model
model = joblib.load("notice_classifier.joblib")

# Test examples: title, content, expected category
test_notices = [
    (
        "Campus Placement Drive",
        "TCS invites eligible final-year students to register for its recruitment process.",
        "Placement"
    ),
    (
        "Semester Examination Schedule",
        "University examinations will begin on Monday. Students must check the timetable.",
        "Examination"
    ),
    (
        "Annual Sports Tournament",
        "Students can register for the college cricket and football championships.",
        "Sports"
    ),
    (
        "Technical Project Exhibition",
        "Students are invited to showcase innovative engineering projects at the college exhibition.",
        "Event"
    ),
    (
        "Diwali Holiday Notice",
        "The college will remain closed on account of the Diwali festival.",
        "Holiday"
    ),
    (
        "Merit Scholarship Applications",
        "Eligible students can apply for the merit-based scholarship before the deadline.",
        "Scholarship"
    ),
    (
        "Open Elective Selection",
        "Students must select their preferred elective subjects through the academic portal.",
        "Academics"
    ),
    (
        "Student Identity Card Update",
        "Students must visit the administrative office to update their identity card details.",
        "General"
    ),
    (
        "Public Speaking Competition",
        "Students are invited to participate in the interdepartmental debate and public speaking contest.",
        "Event"
    ),
    (
        "Intercollege Chess Tournament",
        "Interested students can register for the intercollege chess championship.",
        "Sports"
    ),
    (
        "Hall Ticket Verification",
        "Candidates must verify their name, photograph, and examination details on their hall tickets.",
        "Examination"
    ),
    (
        "Campus Hiring Interview Results",
        "Shortlisted candidates will receive their final interview results from the placement cell.",
        "Placement"
    ),
    (
        "Government Holiday Announcement",
        "The institution will remain closed on the officially declared public holiday.",
        "Holiday"
    ),
    (
        "Research Fellowship Notice",
        "Eligible postgraduate researchers are invited to apply for the institutional research fellowship.",
        "Scholarship"
    ),
    (
        "Web Development Bootcamp",
        "Join the coding club for a hands-on workshop covering HTML, CSS, JavaScript, and web development.",
        "Event"
    ),
    (
        "Football Team Selection Trials",
        "Students must attend football trials for selection to the college sports team.",
        "Sports"
    )
]

print("=" * 75)
print("NOTICE CLASSIFIER TEST RESULTS")
print("=" * 75)

correct = 0

for title, content, expected in test_notices:
    text = title + " " + content
    predicted = model.predict([text])[0]

    result = "PASS" if predicted == expected else "CHECK"

    if predicted == expected:
        correct += 1

    print(f"\nTitle:    {title}")
    print(f"Expected: {expected}")
    print(f"Predicted: {predicted}")
    print(f"Result:   {result}")

print("\n" + "=" * 75)
print(f"Correct: {correct}/{len(test_notices)}")
print(f"Test accuracy: {correct / len(test_notices) * 100:.2f}%")
print("=" * 75)
