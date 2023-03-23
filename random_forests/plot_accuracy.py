import matplotlib.pyplot as plt
def plot_acc(feature_range, test_scores, oob_scores, train_scores):
    plt.figure(figsize=(12, 7))

    plt.plot(feature_range, test_scores, label="test scores")
    plt.plot(feature_range, oob_scores, label="oob scores")
    plt.plot(feature_range, train_scores, label="train scores")
    plt.legend(fontsize=20, loc='best')
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.ylabel("accuracy", fontsize=20)
    plt.xlabel("max_features", fontsize=20)
    plt.show()


