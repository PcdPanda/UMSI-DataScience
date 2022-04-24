import pandas as pd

if __name__ == "__main__":
    result = pd.read_csv("pandapcd_zyzhe_si618_hw7_batch_result.csv").set_index(["Input.comment_text"])
    toxic = result[["Answer.Question 1 - Yes .toxic1.yes"]].rename(columns={"Answer.Question 1 - Yes .toxic1.yes": "yes"})
    toxic = toxic.groupby(level=0).sum()
    comments = toxic[toxic["yes"]>=2].index
    targeted = result[["Answer.Question 2 - Yes .target2.yes"]].rename(columns={"Answer.Question 2 - Yes .target2.yes": "targeted"}).loc[comments].groupby(level=0).sum()
    with open("pandapcd_zyzhe_si618_hw7_compute.txt", "w") as f:
        f.write("toxic_1\t{}\n".format(len(toxic[toxic["yes"]>=1]) / len(toxic)))
        f.write("toxic_2\t{}\n".format(len(toxic[toxic["yes"]>=2]) / len(toxic)))
        f.write("toxic_3\t{}\n".format(len(toxic[toxic["yes"]>=3]) / len(toxic)))
        f.write("targeted\t{}\n".format(len(targeted[targeted["targeted"] >= 1]) / len(toxic)))