from sklearn.datasets import load_svmlight_file;
from sklearn.ensemble import RandomForestClassifier
import csv
import numpy as np
import random
import sys
from sklearn import cross_validation
from sklearn.metrics import auc_score, accuracy_score, precision_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier

data = []
label = []

####################################################
# load letter dataset
f = open('../UCI_dataset/letter-recognition.data', 'r')
csv_reader = csv.reader(f)

for row in csv_reader:
	t = []
	for i in range(16):
		t.append(row[i + 1])
	if row[0] == 'O':
		ltemp = 1
	else:
		ltemp = -1
	t.append(ltemp)
	data.append(t)

data = np.array(data)

random.shuffle(data)

ndata = []
label = []

for row in data:
	ndata.append(row[0:16])
	label.append(int(row[16]))

data = np.array(ndata)
label = np.array(label)

test_data = data[5000:19000]
test_label = label[5000:19000]

data = data[0:5000]
label = label[0:5000]

print len(data), len(label), len(test_data), len(test_label)


####################################################
####################################################
kf = cross_validation.KFold(5000, n_folds=5)

niter = [2,4,8,16,32,64,128,256,512,1024]
# niter = [2,4]
count_cv = 0
total_score = 0
total_cv_score = 0

te_roc = 0
te_acc = 0
te_apr = 0
te_f1 = 0

cv_roc = 0
cv_acc = 0
cv_apr = 0
cv_f1 = 0

for train_index, test_index in kf:
	count_cv = count_cv + 1
	print '5-fold CV No.', count_cv

	best_clf = None
	max_score = 0
	total = 0
	best_roc = 0
	best_acc = 0
	best_apr = 0
	best_f1 = 0
	best_nf = 0
	
	# print("TRAIN:", train_index, "TEST:", test_index)
	X_train, X_test = data[train_index], data[test_index]
	y_train, y_test = label[train_index], label[test_index]
	
	# choose best parameter
	# test linear kernel
	for nf in niter:
		clf = GradientBoostingClassifier(n_estimators=nf)
		clf.fit(X_train, y_train)
		pred = clf.predict(X_test)

		roc = auc_score(y_test, pred)
		acc = accuracy_score(y_test, pred)
		apr = precision_score(y_test, pred)
		f1 = f1_score(y_test, pred)
	
		total = roc + acc + apr + f1
		print '5-fold CV No.', count_cv, ', nf: ', nf, 'total_score=', total

		if total > max_score:
			max_score = total
			best_clf = clf
			best_roc = roc
			best_acc = acc
			best_apr = apr
			best_f1 = f1
			best_nf = nf

	print '5-fold CV No.', count_cv, ', best score(CV) = ', (max_score / 4), ', nf: ', best_nf, 'best acc, roc, apr, f1(CV)=', best_acc, best_roc, best_apr, best_f1
	
	cv_roc = cv_roc + best_roc
	cv_acc = cv_acc + best_acc
	cv_apr = cv_apr + best_apr
	cv_f1 = cv_f1 + best_f1

	total_cv_score  = total_cv_score + (max_score / 4)

	# try best parameter on big test set
	test_pred = best_clf.predict(test_data)

	test_roc = auc_score(test_label, test_pred)
	test_acc = accuracy_score(test_label, test_pred)
	test_apr = precision_score(test_label, test_pred)
	test_f1 = f1_score(test_label, test_pred)
	avg = (test_acc + test_roc + test_apr + test_f1) / 4

	print '5-fold CV No.', count_cv, 'test score: acc, roc, apr, f1=', test_acc, test_roc, test_apr, test_f1, ', avg= ', avg
	total_score = total_score + avg

	te_roc = te_roc + test_roc
	te_acc = te_acc + test_acc
	te_apr = te_apr + test_apr
	te_f1 = te_f1 + test_f1

print 'total cv avg=', (total_cv_score / count_cv)
print 'cv avg acc, roc, apr, f1=', (cv_acc / count_cv), (cv_roc / count_cv), (cv_apr / count_cv), (cv_f1 / count_cv)

print 'total test avg=', (total_score / count_cv)
print 'test avg acc, roc, apr, f1=', (te_acc / count_cv), (te_roc / count_cv), (te_apr / count_cv), (te_f1 / count_cv)

#####################################################
# letter p1, train:test 4000:1000, feature=10, acc=92.9%
# letter p1, train:test 4000:1000, feature=10, acc=98.7%
# letter p1, train:test 4000:1000, feature=1, acc=98.2%

# letter.p1, 10-fold CV, feature=10, 0.9376
# letter.p2, 10-fold CV, feature=10, 0.987