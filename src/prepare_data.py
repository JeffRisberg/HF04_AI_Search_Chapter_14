import pandas
from datasets import Dataset, DatasetDict

path = ""

def getTrainingData(filename):
    golden_answers = pandas.read_csv(filename)
    golden_answers["class"] = golden_answers["class"].fillna(-2).astype(int)
    validated = golden_answers[golden_answers["class"] > -1]

    table = {"id": [],
             "url": [],
             "title": [],
             "question": [],
             "context": [],
             "answers": []}

    for idx, row in validated.iterrows():
        answers = row["gold"].split('|')
        starts = []
        notfound = False
        for i in range(len(answers)):
            found = row["context"].find(answers[i])
            starts.append(found)
            if (found < 0):
                notfound = True
        if not notfound:
            table["id"].append(row["id"])
            table["url"].append(row["url"])
            table["title"].append(row["question"])
            table["question"].append(row["question"])
            table["context"].append(row["context"])
            table["answers"].append({
                "text": answers,
                "answer_start": starts
            })
    df = pandas.DataFrame(table).sample(frac=1)

    train_split = int(len(df) * 0.75)
    eval_split = int((len(df) - train_split) / 1.25) + train_split - 1

    train_dataset = Dataset.from_pandas(df[:train_split])
    test_dataset = Dataset.from_pandas(df[train_split:eval_split])

    validation_dataset = Dataset.from_pandas(df[eval_split:])

    # datasets is a utility class in HuggingFace
    datadict = DatasetDict({'train': train_dataset,
                            'test': test_dataset,
                            'validation': validation_dataset})

    return datadict


datadict = getTrainingData(path + 'outdoors_golden_answers.csv')
datadict.save_to_disk(path + 'data/question-answering-training-set')
