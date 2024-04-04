import json
from collections import OrderedDict

data = [
    # {"ClassId": "rice", "SearchStrings": ["bag of white rice", "box of rice"]},
    # {
    #     "ClassId": "red wine",
    #     "SearchStrings": ["red wine", "red wine bottle", "red wine bottle table"],
    # },
    # {"ClassId": "corn", "SearchStrings": ["corn", "corn on the cob", "raw corn"]},
    # {
    #     "ClassId": "cabbage",
    #     "SearchStrings": ["cabbage", "cabbage leaves", "raw cabbage"],
    # },
    # {"ClassId": "egg", "SearchStrings": ["eggs", "eggs in shells", "eggs in carton"]},
    # {
    #     "ClassId": "bell pepper",
    #     "SearchStrings": ["bell pepper", "raw bell pepper", "fresh bell peppers"],
    # },
    # {
    #     "ClassId": "cream cheese",
    #     "SearchStrings": [
    #         "cream cheese",
    #         "cream cheese box",
    #         "philadelphia cream cheese",
    #         "philadelphia cream cheese container",
    #     ],
    # },
    # {"ClassId": "cream", "SearchStrings": ["heavy whipping cream", "heavy cream"]},
    # {
    #     "ClassId": "bread crumbs",
    #     "SearchStrings": [
    #         "bread crumb packaging",
    #         "bread crumbs packaging",
    #     ],
    # },
    # {
    #     "ClassId": "mushroom soup",
    #     "SearchStrings": ["mushroom soup can", "mushroom soup cans"],
    # },
    # {
    #     "ClassId": "cream ofchicken soup",
    #     "SearchStrings": ["cream of chicken soup cans", "cream of chicken soup can"],
    # },
    # {
    #     "ClassId": "tomato paste",
    #     "SearchStrings": ["tomato paste can", "tomato paste cans"],
    # },
    # {
    #     "ClassId": "sesame oil",
    #     "SearchStrings": ["sesame oil", "sesame oil bottle", "sesame oil bottles"],
    # },
    # {
    #     "ClassId": "buttermilk",
    #     "SearchStrings": [
    #         "buttermilk container",
    #         "buttermilk carton",
    #         "buttermilk cartons",
    #     ],
    # },
    # {
    #     "ClassId": "black beans",
    #     "SearchStrings": ["black beans can", "black beans cans"],
    # },
    # {"ClassId": "chives", "SearchStrings": ["fresh chives", "fresh chives in grocery"]},
    # {
    #     "ClassId": "salsa",
    #     "SearchStrings": ["jar of salsa", "jars of salsa", "salsa jar in grocery"],
    # },
    # {
    #     "ClassId": "cottage cheese",
    #     "SearchStrings": ["cottage cheese container", "cottage cheese containers"],
    # },
    # {
    #     "ClassId": "onion soup",
    #     "SearchStrings": [
    #         "onion soup can",
    #         "onion soup cans",
    #         "onion soup packet",
    #         "onion soup packaging",
    #     ],
    # },
    # {
    #     "ClassId": "capers",
    #     "SearchStrings": [
    #         "jar of capers",
    #         "jars of capers",
    #         "jars of capers",
    #         "capers in grocery store",
    #     ],
    # },
    # {
    #     "ClassId": "water chestnuts",
    #     "SearchStrings": [
    #         "water chestnuts can",
    #         "water chestnuts cans",
    #         "water chestnuts can grocery",
    #     ],
    # },
    # {
    #     "ClassId": "horseradish",
    #     "SearchStrings": [
    #         "horseradish jar",
    #         "horseradish jars",
    #         "horseradish jars grocery",
    #     ],
    # },
    # {
    #     "ClassId": "sherry",
    #     "SearchStrings": [
    #         "cooking sherry",
    #         "cooking sherry grocery",
    #         "cooking sherry bottle",
    #     ],
    # },
    # {
    #     "ClassId": "molasses",
    #     "SearchStrings": ["molasses jar", "molasses jars", "molasses jar grocery"],
    # },
    # {"ClassId": "tuna", "SearchStrings": ["tuna can", "tuna cans", "tuna can grocery"]},
    # {
    #     "ClassId": "pork chops",
    #     "SearchStrings": ["pork chops in packaging", "pork chops in packaging grocery"],
    # },
    # {
    #     "ClassId": "active dry yeast",
    #     "SearchStrings": [
    #         "active dry yeast",
    #         "active dry yeast packets",
    #         "active dry yeast jar",
    #     ],
    # },
    # {
    #     "ClassId": "cornmeal",
    #     "SearchStrings": ["cornmeal box", "cornmeal box grocery", "cornmeal package"],
    # },
    #
    #    Abover are done.. Below are remaining
    # {"ClassId": "green_onion", "SearchStrings": ["green onions", "raw green onions"]},
    # {"ClassId": "celery", "SearchStrings": ["fresh celery", "fresh celery hearts", "raw celery"]},
    # {
    # "ClassId": "ground beef",
    # "SearchStrings": ["ground beef in package", "ground beef in package grocery"],
    # },
    # {"ClassId": "mushroom", "SearchStrings": ["button mushrooms in container"]},
    # {"ClassId": "bacon", "SearchStrings": ["bacon in package"]},
    # {"ClassId": "lime", "SearchStrings": ["fresh limes"]},
    # {"ClassId": "red onion", "SearchStrings": ["fresh red onions"]},
    # {"ClassId": "pecans", "SearchStrings": ["pecans in packaging", "bag of pecans"]},
    # {"ClassId": "shrimp", "SearchStrings": ["shrimp in package", "bag of shrimp"]},
    # {"ClassId": "shallot", "SearchStrings": ["fresh shallot"]},
    # {"ClassId": "orange juice", "SearchStrings": ["orange juice in container"]},
    # {"ClassId": "broccoli", "SearchStrings": ["bunch of broccoli"]},
    # {"ClassId": "walnuts", "SearchStrings": ["bag of walnuts", "package of walnuts"]},
    # {"ClassId": "raisins", "SearchStrings": ["bag of raisins"]},
    # {"ClassId": "tortilla", "SearchStrings": ["package of tortillas"]},
    # {
    # "ClassId": "green beans",
    # "SearchStrings": ["package of green beans", "fresh green beans"],
    # },
    #    { "ClassId": "onion", "SearchStrings": ["onion", "onions", "raw onions"]},
    #    { "ClassId": "garlic", "SearchStrings": ["garlic", "garlic cloves", "raw garlic"]},
    #    { "ClassId": "tomato", "SearchStrings": ["tomato", "tomatoes", "raw tomatoes"]},
    #    { "ClassId": "potato", "SearchStrings": ["potato", "potatoes", "raw potatoes"]},
    #    { "ClassId": "bell_pepper", "SearchStrings": ["bell pepper", "bell peppers", "raw bell peppers"]},
    #    { "ClassId": "lettuce", "SearchStrings": ["lettuce", "lettuce leaves", "raw lettuce"]},
    #    { "ClassId": "spinach", "SearchStrings": ["spinach", "spinach leaves", "raw spinach"]},
    #    { "ClassId": "kale", "SearchStrings": ["kale", "kale leaves", "raw kale"]},
    #    { "ClassId": "broccoli", "SearchStrings": ["broccoli", "broccoli florets", "raw broccoli"]},
    #    { "ClassId": "cauliflower", "SearchStrings": ["cauliflower", "cauliflower florets", "raw cauliflower"]},
    #    { "ClassId": "asparagus", "SearchStrings": ["asparagus", "asparagus spears", "raw asparagus"]},
    #   { "ClassId": "eggplant", "SearchStrings": ["eggplant", "eggplant slices", "raw eggplant"]},
    #   { "ClassId": "mushroom", "SearchStrings": ["mushroom", "mushrooms", "raw mushrooms"]},
    #   { "ClassId": "green_beans", "SearchStrings": ["green beans", "green beans", "raw green beans"]},
    #    { "ClassId": "green_peas", "SearchStrings": ["green peas", "green peas", "raw green peas"]},
    {
        "ClassId": "chicken",
        "SearchStrings": [
            "raw chicken breast",
            "raw chicken thighs",
            "raw chicken wings",
            "raw_chicken_legs",
        ],
    },
]

json_data = json.dumps(data, indent=2)
with open("queries.json", "w") as f:
    f.write(json_data)
    f.close()
