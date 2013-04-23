from sklearn.datasets import load_svmlight_file;
from sklearn.ensemble import RandomForestClassifier
import csv
import numpy as np
import random
import sys
from sklearn import cross_validation
from sklearn.metrics import auc_score, accuracy_score, precision_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from pprint import pprint
data = []
label = []

####################################################
# load letter dataset
fin1 = open('../UCI_dataset/covtype_30000_data_scale', 'r')
fin2 = open('../UCI_dataset/covtype_30000_label_scale', 'r')
data = np.load(fin1)
label = np.load(fin2)
fin1.close()
fin2.close()

data = data[0:5000]
label = label[0:5000]

print len(data), len(label)


####################################################
####################################################
bs = cross_validation.Bootstrap(5000, n_iter=30, train_size=4000, test_size=1000)

rad = [0.001,0.005,0.01,0.05,0.1,0.5,1,2]
deg = [2,3]
n_f = [1,2,4,6,8,12,16,20]
niter = [2,4,8,16,32,64,128,256,512,1024]

count_cv = 0
svm_rank = []
rf_rank = []
boost_rank = []

for train_index, test_index in bs:
	count_cv = count_cv + 1
	print 'bootstrap', count_cv

	svm_score = 0
	rf_score = 0
	boost_score = 0
	
	# print("TRAIN:", train_index, "TEST:", test_index)
	X_train, X_test = data[train_index], data[test_index]
	y_train, y_test = label[train_index], label[test_index]

	# test svm
	best_clf = None
	max_score = 0
	total = 0
	best_roc = 0
	best_acc = 0
	best_apr = 0
	best_f1 = 0
	best_set = None

	# choose best parameter
	# test linear kernel
	for d in deg:
		clf = SVC(kernel='poly', degree=d)
		svm_set = 'poly, degree=' + str(d)
		clf.fit(X_train, y_train)
		pred = clf.predict(X_test)

		roc = auc_score(y_test, pred)
		acc = accuracy_score(y_test, pred)
		apr = precision_score(y_test, pred)
		f1 = f1_score(y_test, pred)
	
		total = roc + acc + apr + f1
		print 'bootstrap No.', count_cv, ', setting: ', svm_set, 'total_score=', total

		if total > max_score:
			max_score = total
			best_clf = clf
			best_roc = roc
			best_acc = acc
			best_apr = apr
			best_f1 = f1
			best_set = svm_set

	# test linear kernel
	clf = SVC(kernel='linear')
	svm_set = 'linear'
	clf.fit(X_train, y_train)
	pred = clf.predict(X_test)

	roc = auc_score(y_test, pred)
	acc = accuracy_score(y_test, pred)
	apr = precision_score(y_test, pred)
	f1 = f1_score(y_test, pred)

	total = roc + acc + apr + f1
	print 'bootstrap No.', count_cv, ', setting: ', svm_set,', total_score=', total

	if total > max_score:
		max_score = total
		best_clf = clf
		best_roc = roc
		best_acc = acc
		best_apr = apr
		best_f1 = f1
		best_set = svm_set

	# test rbf kernel
	for r in rad:
		clf = SVC(kernel='rbf', gamma=r)
		svm_set = 'rbf, gamma=' + str(r)
		clf.fit(X_train, y_train)
		pred = clf.predict(X_test)

		roc = auc_score(y_test, pred)
		acc = accuracy_score(y_test, pred)
		apr = precision_score(y_test, pred)
		f1 = f1_score(y_test, pred)
	
		total = roc + acc + apr + f1
		print 'bootstrap No.', count_cv, ', setting: ', svm_set, ', total_score=', total

		if total > max_score:
			max_score = total
			best_clf = clf
			best_roc = roc
			best_acc = acc
			best_apr = apr
			best_f1 = f1
			best_set = svm_set

	svm_score = max_score

	print 'bootstrap No.', count_cv, ', best svm_score = ', (svm_score / 4)

	# test boost
	best_clf = None
	max_score = 0
	total = 0
	best_roc = 0
	best_acc = 0
	best_apr = 0
	best_f1 = 0
	best_nf = 0
	
	# choose best parameter
	for nf in niter:
		clf = GradientBoostingClassifier(n_estimators=nf)
		clf.fit(X_train, y_train)
		pred = clf.predict(X_test)

		roc = auc_score(y_test, pred)
		acc = accuracy_score(y_test, pred)
		apr = precision_score(y_test, pred)
		f1 = f1_score(y_test, pred)
	
		total = roc + acc + apr + f1
		print 'bootstrap No.', count_cv, ', nf: ', nf, 'total_score=', total

		if total > max_score:
			max_score = total
			best_clf = clf
			best_roc = roc
			best_acc = acc
			best_apr = apr
			best_f1 = f1
			best_nf = nf

	boost_score = max_score
	print 'bootstrap No.', count_cv, ', best boost score = ', (boost_score / 4)

	# test rf
	best_clf = None
	max_score = 0
	total = 0
	best_nf = 0

	best_roc = 0
	best_acc = 0
	best_apr = 0
	best_f1 = 0
	
	# choose best parameter
	for i in range(len(n_f)):
		clf = RandomForestClassifier(n_estimators=1024, max_features=n_f[i])
		clf.fit(X_train, y_train)
		pred = clf.predict(X_test)

		roc = auc_score(y_test, pred)
		acc = accuracy_score(y_test, pred)
		apr = precision_score(y_test, pred)
		f1 = f1_score(y_test, pred)
	
		total = roc + acc + apr + f1
		print 'bootstrap No.', count_cv, ', max_feature=', n_f[i], ', total_score=', total

		if total > max_score:
			max_score = total
			best_clf = clf
			best_nf = n_f[i]
			best_roc = roc
			best_acc = acc
			best_apr = apr
			best_f1 = f1
	rf_score = max_score
	print 'bootstrap No.', count_cv, ', best rf score = ', (rf_score / 4)

	all_score = [svm_score, boost_score, rf_score]
	sort_all = [i[0] for i in sorted(enumerate(all_score), key=lambda x:x[1])]
	svm_rank.append(sort_all[0] + 1)
	boost_rank.append(sort_all[1] + 1)
	rf_rank.append(sort_all[2] + 1)

pprint(svm_rank)
pprint(rf_rank)
pprint(boost_rank)
fout=open('./result_cov.txt', 'w')
print >> fout, svm_rank.count(1), svm_rank.count(2), svm_rank.count(3)
print >> fout, boost_rank.count(1), boost_rank.count(2), boost_rank.count(3)
print >> fout, rf_rank.count(1), rf_rank.count(2), rf_rank.count(3)