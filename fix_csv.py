import csv

INPUT_FILE = "aicommerce_data.csv"
OUTPUT_FILE = "aicommerce_data_fixed.csv"

expected_columns = 6
fixed_rows = []

with open(INPUT_FILE, "r", encoding="utf-8") as infile:
    reader = csv.reader(infile)
    header = next(reader)
    fixed_rows.append(header)

    for i, row in enumerate(reader, start=2):
        if len(row) == expected_columns:
            fixed_rows.append(row)
        elif len(row) > expected_columns:
            # מאחד את כל ההתחלה לעמודת name
            fixed_name = ",".join(row[:len(row) - 5])
            fixed_row = [fixed_name] + row[-5:]
            fixed_rows.append(fixed_row)
            print(f"⚠️ תוקנה שורה {i}: פסיקים עודפים בשם מוצר")
        else:
            print(f"❌ דולגה שורה {i} – מעט מדי עמודות: {row}")

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as outfile:
    writer = csv.writer(outfile)
    writer.writerows(fixed_rows)

print("\n✅ הסתיים תיקון הקובץ!")
print("נוצר קובץ חדש תקין בשם:")
print("👉 aicommerce_data_fixed.csv")
print("\nעכשיו:")
print("1️⃣ מחק את aicommerce_data.csv הישן")
print("2️⃣ שנה שם ל־aicommerce_data_fixed.csv → aicommerce_data.csv")
print("3️⃣ הפעל שוב את האתר ✅")
