class U_trans_model():
    """Apply unsupervised transformer prior to the model,
    Input example:
    U_trans_model(svm.SVC(kernel='rbf', probability=True), transformer=PCA(n_components=1))"""
  
    def __init__(self, model, transformer):
        self.transformer = transformer
        self.clf = model

    def fit(self, X, y):
        X = self.transformer.fit_transform(X)
        self.clf.fit(X, y)

    def predict(self, X_t):
        X_t = self.transformer.transform(X_t)
        return self.clf.predict(X_t)

    def predict_proba(self, X_t):
        X_t = self.transformer.transform(X_t)
        return self.clf.predict_proba(X_t)

    def score(self, X_t, y_t):
        X_t = self.transformer.transform(X_t)
        return self.clf.score(X_t, y_t)


class S_trans_model():
    """Apply supervised transformer prior to the model,
    Input example:
    S_trans_model(svm.SVC(kernel='rbf', probability=True),LinearDiscriminantAnalysis(n_components=1))"""
  
    def __init__(self, model, transformer):
        self.transformer = transformer
        self.clf = model

    def fit(self, X, y):
        X = self.transformer.fit_transform(X, y)
        self.clf.fit(X, y)

    def predict(self, X_t):
        X_t = self.transformer.transform(X_t)
        return self.clf.predict(X_t)

    def predict_proba(self, X_t):
        X_t = self.transformer.transform(X_t)
        return self.clf.predict_proba(X_t)

    def score(self, X_t, y_t):
        X_t = self.transformer.transform(X_t)
        return self.clf.score(X_t, y_t)