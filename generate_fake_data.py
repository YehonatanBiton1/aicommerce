import csv
import random
from datetime import datetime

categories = ["טכנולוגיה", "רכב", "כושר", "בית", "גאדג'טים"]

with open("aicommerce_data.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["email","name","category","price","trend_score","real_success","created_at"])

    for i in range(30):
        writer.writerow([
            "demo@gmail.com",
            f"Product {i+1}",
            random.choice(categories),
            random.randint(20, 200),
            random.randint(30, 100),
            random.choice([0, 1]),
            datetime.now().isoformat()
        ])

print("✅ נוצר דאטה פיקטיבי לאימון מודל")
