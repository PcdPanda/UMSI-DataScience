import json

with open("line.json", "r") as f:
    line = json.loads(f.read())


def mapper(line):
    ans = list()
    city = line.get("city", None)
    try:
        categories = line["categories"].split(", ")
    except Exception:
        categories = [""]
    num = 1
    review_count = line["review_count"]
    rating = line["stars"]
    try:
        assert line["attributes"]["WheelchairAccessible"] == "True"
        wheel = 1
    except Exception:
        wheel = 0
    parking = 0
    for key in ["garage", "street", "lot"]:
        try:
            v = eval(line["attributes"]["BusinessParking"])[key]
            if v in ["True", True]:
                parking = 1
                break
        except Exception:
            pass
    for category in categories:
        ans.append([(city, category), (parking, wheel, num, rating, review_count)])
    return ans


print(mapper(line))
