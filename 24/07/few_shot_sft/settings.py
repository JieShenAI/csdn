LABELS_DICT = {
    0: "Human Necessities",
    1: "Performing Operations; Transporting",
    2: "Chemistry; Metallurgy",
    3: "Textiles; Paper",
    4: "Fixed Constructions",
    5: "Mechanical Engineering; Lightning; Heating; Weapons; Blasting",
    6: "Physics",
    7: "Electricity",
    8: "General tagging of new or cross-sectional technology",
}

LABELS_NAME = [
    LABELS_DICT[i]
    for i in range(9)
]

LABELS_2_IDS = {
    v : k
    for k, v in LABELS_DICT.items()
}

categories = []
for i in range(len(LABELS_DICT)):
    categories.append(f'\"{LABELS_DICT[i]}\"')
categories += ['\"Unknown\"']


LLM_CLS_FORMAT = """
You are a document classifier. When given the text, you classify the text into one of the following categories:

{categories}

Your output should only contain one of the categories and no explanation or any other text.
""".strip()

HUMAN_FORMAT = "Classify the document:\n{input}"

if __name__ == "__main__":
    print(LLM_CLS_FORMAT.format(categories="\n".join(categories)))