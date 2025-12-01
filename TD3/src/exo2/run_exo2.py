import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

header = ["Customer ID", "Gender", "Car Type", "Shirt Size", "Class"]
rows = [
    ["1",  "M", "Family",      "Small",       "C0"],
    ["2",  "M", "Sports",      "Medium",      "C0"],
    ["3",  "M", "Sports",      "Medium",      "C0"],
    ["4",  "M", "Sports",      "Large",       "C0"],
    ["5",  "M", "Sports",      "Extra Large", "C0"],
    ["6",  "M", "Sports",      "Extra Large", "C0"],
    ["7",  "F", "Sports",      "Small",       "C0"],
    ["8",  "F", "Sports",      "Small",       "C0"],
    ["9",  "F", "Sports",      "Medium",      "C0"],
    ["10", "F", "Luxury",      "Large",       "C0"],
    ["11", "M", "Family",      "Large",       "C1"],
    ["12", "M", "Family",      "Extra Large", "C1"],
    ["13", "M", "Family",      "Medium",      "C1"],
    ["14", "M", "Luxury",      "Extra Large", "C1"],
    ["15", "F", "Luxury",      "Small",       "C1"],
    ["16", "F", "Luxury",      "Small",       "C1"],
    ["17", "F", "Luxury",      "Medium",      "C1"],
    ["18", "F", "Luxury",      "Medium",      "C1"],
    ["19", "F", "Luxury",      "Medium",      "C1"],
    ["20", "F", "Luxury",      "Large",       "C1"],
]

dataset = (header, rows)

from impurity_split import test_impurity_split
from helpers import gini_dataset, gini_attribute_multiway, gini_attribute_binary_splits

def run_exercise_2():
    test_impurity_split()

    print("2(b) Gini of entire dataset:")
    print("  Gini(dataset) =", gini_dataset(dataset))       # should be 0.5
    print()

    print("2(c) Gini of attribute Customer ID (multiway):")
    g_customer, _ = gini_attribute_multiway(dataset, "Customer ID")
    print("  Gini(Customer ID) =", g_customer)              # 0.0
    print()

    print("2(d) Gini of attribute Gender (multiway):")
    g_gender, _ = gini_attribute_multiway(dataset, "Gender")
    print("  Gini(Gender) =", g_gender)                     # ≈ 0.48
    print()

    print("2(e) Gini of attribute Car Type:")
    g_car_multi, _ = gini_attribute_multiway(dataset, "Car Type")
    print("  Multiway split:", g_car_multi)                 # ≈ 0.1625

    print("  Binary splits:")
    for subset, g_split in gini_attribute_binary_splits(dataset, "Car Type"):
        print(f"    {subset} vs others -> Gini_split = {g_split:.3f}")
    print()

    print("2(f) Gini of attribute Shirt Size (multiway):")
    g_shirt, _ = gini_attribute_multiway(dataset, "Shirt Size")
    print("  Gini(Shirt Size) =", g_shirt)                  # ≈ 0.491
    print()

    print("2(g) Best attribute among Gender / Car Type / Shirt Size?")
    print("  Gini(Gender)      =", g_gender)
    print("  Gini(Car Type)    =", g_car_multi)
    print("  Gini(Shirt Size)  =", g_shirt)
    print("  -> Best (lowest Gini): Car Type")
    print()

    print("2(h) Why not use Customer ID although Gini = 0?")
    print("  Because Customer ID is just an identifier;")
    print("  the split memorizes each training example (overfitting)")
    print("  and does not generalize to new customers.")


if __name__ == "__main__":
    run_exercise_2()
