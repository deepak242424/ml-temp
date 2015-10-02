import numpy as np
import sklearn.linear_model as LM
from sklearn.linear_model import Ridge
from sklearn.grid_search import GridSearchCV

def linear_regression_over_splits(splits, prefix):
    rss = []
    for cnt in range(splits):
        fname_x = prefix + '-train0' + str(cnt+1) + '.csv'
        fname_y = prefix + '-test0' + str(cnt+1) + '.csv'
        trdata = np.genfromtxt(fname_x, delimiter = ',')
        testdata = np.genfromtxt(fname_y, delimiter = ',')

        tr_X = trdata[:,:-1]
        tr_Y = trdata[:,-1]
        te_X = testdata[:,:-1]
        te_Y = testdata[:,-1]

        clf_LR = LM.LinearRegression()
        clf_LR.fit(tr_X, tr_Y)
        rss.append(np.mean((clf_LR.predict(te_X)- te_Y) ** 2))
        print 'RSS for Split ', cnt+1, ':', np.mean((clf_LR.predict(te_X)- te_Y) ** 2)
        #print 'Coefficients Learned : ', clf_LR.coef_
    print 'Mean RSS : ', np.mean(rss)

def ridge_regression(splits, prefix):
    for alp_val in range(10):
        alp_val *= 100
        rss = []
        for cnt in range(splits):
            fname_x = prefix + '-train0' + str(cnt+1) + '.csv'
            fname_y = prefix + '-test0' + str(cnt+1) + '.csv'
            trdata = np.genfromtxt(fname_x, delimiter = ',')
            testdata = np.genfromtxt(fname_y, delimiter = ',')

            tr_X = trdata[:,:-1]
            tr_Y = trdata[:,-1]
            te_X = testdata[:,:-1]
            te_Y = testdata[:,-1]

            clf_LR = LM.Ridge(alpha=alp_val, normalize=True)
            clf_LR.fit(tr_X, tr_Y)
            rss.append(np.mean((clf_LR.predict(te_X)- te_Y) ** 2))
            #print 'RSS for Split ', cnt+1, ':', np.mean((clf_LR.predict(te_X)- te_Y) ** 2)
            #print 'Coefficients Learned : ', clf_LR.coef_

            print 'Mean RSS : ', np.mean(rss)
'''
def ridge_regression(splits, prefix):
        for cnt in range(splits):
            fname_x = prefix + '-train0' + str(cnt+1) + '.csv'
            fname_y = prefix + '-test0' + str(cnt+1) + '.csv'
            trdata = np.genfromtxt(fname_x, delimiter = ',')
            testdata = np.genfromtxt(fname_y, delimiter = ',')

            tr_X = trdata[:,:-1]
            tr_Y = trdata[:,-1]
            te_X = testdata[:,:-1]
            te_Y = testdata[:,-1]
            alphas = np.array([955,10,0.1,0.01,0.001,0.0001,0])
            #alphas = np.asarray(xrange[-10, 10, 1])
            model = Ridge()
            grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))
            grid.fit(tr_X, tr_Y)
            #print(grid.best_score_)
            print(grid.best_estimator_.alpha)
        #print 'Mean RSS : ', np.mean(rss)
'''

num_splits = 5
linear_regression_over_splits(num_splits, prefix='CandC')
linear_regression_over_splits(num_splits, prefix='Reduced')
#ridge_regression(splits=num_splits, prefix='CandC')
#print '*************************************'
#ridge_regression(splits=num_splits, prefix='Reduced')




