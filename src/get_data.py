import pandas

def getQuestions():
    qtypes = ['Who','What','When','Where','Why','How']
    questions = []
    for qt in qtypes:
        lq = len(qt)
        request = {
            "query": qt,
            "fields": ["id", "url", "owner_user_id", "title",
                       "accepted_answer_id"],
            "params": {
                "qf": ["title"],
                "fq": ["accepted_answer_id:[* TO *]"],
                "defType": "edismax",
                "rows":10000
            } }
        docs = requests.post(solr_url + outdoors_collection + "/select",
                             json=request).json()["response"]["docs"]
        questions += [doc for doc in docs if doc['title'][0:lq]==qt]
    return questions

def getContextDataFrame(questions):
    contexts={"id":[],"question":[],"context":[],"url":[]}
    for question in questions:
        request = {
            "query": "*:*",
            "fields": ["body"],
            "params": {
                "fq": ["id:"+str(question["accepted_answer_id"])],
                "defType": "edismax",
                "rows":1,
                "sort":"score desc"
            } }
        docs = requests.post(solr_url + outdoors_collection + "/select",
                             json=request).json()["response"]["docs"]
        contexts["id"].append(question["id"])
        contexts["url"].append(question["url"])
        contexts["question"].append(question["title"]),
        contexts["context"].append(docs[0]["body"])
    return pandas.DataFrame(contexts)
questions = getQuestions()
contexts = getContextDataFrame(questions)
