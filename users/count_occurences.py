import ast
import csv
import matplotlib.pyplot as plt


def get_classes_from_user_features(filepath):
    """
    Parse user_features.py via AST to get class names reliably.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=filepath)

    class_names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_names.append(node.name)
    return class_names


def get_classes_and_bodies_from_benefits_programs(filepath):
    """
    Parse benefits_programs.py via AST to get a dict of
      {class_name: raw_source_code_of_that_class_body}
    """
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source, filename=filepath)
    lines = source.splitlines(True)  # keep line endings

    classes_dict = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_name = node.name
            start = node.lineno - 1  # AST nodes are 1-based
            end = node.end_lineno or start
            class_body_src = "".join(lines[start:end])
            classes_dict[class_name] = class_body_src
    return classes_dict


def main():
    user_features_file = "users/user_features.py"
    benefits_programs_file = "users/benefits_programs.py"

    # 1. Get user_features classes
    user_features_classes = get_classes_from_user_features(user_features_file)

    # 2. Get benefits_programs classes & bodies
    benefits_classes = get_classes_and_bodies_from_benefits_programs(
        benefits_programs_file
    )

    # 3. Build references map: {uf_class: {"count": int, "classes": list_of_bp_classes}}
    references = {
        class_name: {"count": 0, "classes": []} for class_name in user_features_classes
    }

    # 4. Check for substring ["class_name"] in each benefits class body
    for uf_class in user_features_classes:
        pattern_to_find = f'["{uf_class}"]'
        for bp_class_name, bp_class_body in benefits_classes.items():
            if pattern_to_find in bp_class_body:
                references[uf_class]["count"] += 1
                references[uf_class]["classes"].append(bp_class_name)

    # 5. Sort by "Number of referencing classes" descending
    #    sorted_references will be a list of tuples: [(uf_class, {"count":..., "classes":[...]}), ...]
    sorted_references = sorted(
        references.items(), key=lambda x: x[1]["count"], reverse=True
    )

    # 6. Write results to CSV (sorted)
    csv_filename = "references_report.csv"
    with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # Header
        writer.writerow(
            [
                "Class in user_features",
                "Number of referencing classes in benefits_programs",
                "Referencing classes in benefits_programs",
            ]
        )

        for uf_class, info in sorted_references:
            writer.writerow([uf_class, info["count"], ", ".join(info["classes"])])

    print(f"Sorted report generated: {csv_filename}")

    # 7. Plot bar graph (sorted_references)
    sorted_references = sorted_references[:20]  # top 20

    def fix_labels(l):
        l = l.replace("_", " ")
        l = l.replace("place of residence", "city")
        l = l.replace("annual work income", "work income")
        l = l.replace("current school level", "grade")
        l = l.replace("work hours per week", "work h/week")
        l = l.replace("annual investment income", "investment income")
        l = l.replace("primary residence", "residence type")
        l = l.replace("enrolled in educational training", "educational training")
        l = l.replace("receives temporary assistance", "temporary assistance")
        l = l.replace("authorized to work in us", "US work authorization")
        # l = l.replace("is parent", "parent")
        return l

    #    x-axis = user_features class, y-axis = referencing classes count
    class_names = [fix_labels(item[0]) for item in sorted_references]  # uf_class
    counts = [item[1]["count"] for item in sorted_references]
    plt.figure(figsize=(12, 9))
    plt.bar(class_names, counts, color="skyblue")
    plt.xticks(rotation=60, ha="right", fontsize=12)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.ylabel("Dependent Opportunities", fontsize=28)
    plt.tight_layout()  # ensures labels fit nicely

    # Save the figure
    plot_filename = "references_bar_chart.png"
    plt.savefig(plot_filename, dpi=150)
    plt.close()

    print(f"Bar chart saved: {plot_filename}")


if __name__ == "__main__":
    main()

# https://chatgpt.com/c/67a59b9d-aa74-8002-8f32-24d821f44fe4
