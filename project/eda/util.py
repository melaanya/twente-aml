def plot_top_features(model, feature_names, num):
    top = sorted(zip(feature_names, model.feature_importances_),
                  key = lambda x: x[1], reverse = True)[:num]
    top = sorted(top, key = lambda x: x[1])
    plot_top = zip(*top)
    plt.figure()
    plt.barh(range(0, len(plot_top[0])), plot_top[1], tick_label = plot_top[0])
    plt.title('Feature importance')
    plt.show()
    return top


def crossvaltest(params, train_set, train_label, cat_dims, n_splits=3):
    kf = KFold(n_splits=n_splits, shuffle=True) 
    res = []
    for train_index, test_index in kf.split(train_set):
        train = train_set.iloc[train_index,:]
        test = train_set.iloc[test_index,:]

        labels = train_label.ix[train_index]
        test_labels = train_label.ix[test_index]

        clf = CatBoostRegressor(**params)
        clf.fit(train, np.ravel(labels), cat_features=cat_dims)

        res.append(np.mean(clf.predict(test)==np.ravel(test_labels)))
    return np.mean(res)

def catboost_param_tune(params, train_set, train_label,cat_dims=None,n_splits=3):
    ps = paramsearch(params)
    for prms in chain(ps.grid_search(['learning_rate','iterations','depth'])):
        prms['logging_level'] = 'Silent'
        res = crossvaltest(prms,train_set,train_label,cat_dims,n_splits)
        ps.register_result(res,prms)
    return ps.bestparam()

