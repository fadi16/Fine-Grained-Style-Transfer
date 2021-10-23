import pickle

if __name__ == "__main__":
    w2id, id2w = pickle.load(open("w2id_id2w.pkl", "rb"))

    for mode in ["train", "test", "valid"]:
        print(">>>" + mode + "<<<")
        encoded_sentances, classes = pickle.load(open(mode + "_C.pkl", "rb"))
        for i in range(10):
            s = "  ".join(id2w[id] for id in encoded_sentances[i])
            print(s)
            print(classes[i])

        for i in range(10):
            s = "  ".join(id2w[id] for id in encoded_sentances[len(encoded_sentances) - 1 - i])
            print(s)
            print(classes[len(encoded_sentances) - 1 - i])
        print("===========================================")
