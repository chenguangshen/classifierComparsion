from sklearn.datasets import load_svmlight_file;
from sklearn.ensemble import RandomForestClassifier
import csv
import numpy as np
import random
import sys
from sklearn import cross_validation
from sklearn.metrics import auc_score, accuracy_score, precision_score, f1_score
from sklearn.svm import SVC

data = []
label = []

# # #####################################################
# # find largest class index = 2
# f = open('UCI_dataset/covtype.data', 'r')
# csv_reader = csv.reader(f)

# count = [0] * 7
# i = 0
# for row in csv_reader:
# 	if (i % 19 == 0):
# 		print row[54]
# 		index = int(row[54]) - 1
# 		count[index] = count[index] + 1
# 	i = i + 1
# print count

# sys.exit(0)

#####################################################
#random select 30000 data from 581012

# data, label = load_svmlight_file("../UCI_dataset/covtype.libsvm.binary.scale")
# data =  np.array(data.todense())
# label = np.array(label).astype(int)

# ndata = []
# count = 0
# for row in data:
# 	if (count % 19 == 0):
# 		t = []
# 		t.extend(data[count])
# 		if (label[count] == 1):
# 			t.append(1)
# 		else:
# 			t.append(-1)
# 		ndata.append(t)
# 	count = count + 1

# print len(ndata), ndata[0]

# random.shuffle(ndata)

# data = []
# label = []

# for row in ndata:
# 	data.append(row[0:54])
# 	label.append(int(row[54]))

# data = np.array(data)
# label = np.array(label)

# print len(data), len(label), len(data[0]), label[0]

# fi1 = open('../UCI_dataset/covtype_30000_data_scale', 'w')
# fi2 = open('../UCI_dataset/covtype_30000_label_scale', 'w')
# np.save(fi1, data)
# np.save(fi2, label)
# fi1.close()
# fi2.close()

# sys.exit(0)
#####################################################
# load covtype dataset

fin1 = open('../UCI_dataset/covtype_30000_data_scale', 'r')
fin2 = open('../UCI_dataset/covtype_30000_label_scale', 'r')
data = np.load(fin1)
label = np.load(fin2)
fin1.close()
fin2.close()


test_data = data[5000:30000]
test_label = label[5000:30000]

data = data[0:5000]
label = label[0:5000]
print label
# for t in label:
# 	if t!= 1 and t!=2:
# 		print t


print len(data), len(label), len(test_data), len(test_label)

# #####################################################
# # # load letter dataset
# # f = open('UCI_dataset/letter-recognition.data', 'r')
# # csv_reader = csv.reader(f)

# # for row in csv_reader:
# # 	t = []
# # 	for i in range(16):
# # 		t.append(row[i + 1])
# # 	if row[0] == 'O':
# # 		label.append(1)
# # 	else:
# # 		label.append(-1)
# # 	data.append(t)

# # data = numpy.array(data)
# # label = numpy.array(label)
# # #print data

####################################################
####################################################
kf = cross_validation.KFold(5000, n_folds=5)

rad = [0.001,0.005,0.01,0.05,0.1,0.5,1,2]
deg = [2,3]

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
	best_set = None
	
	# print("TRAIN:", train_index, "TEST:", test_index)
	X_train, X_test = data[train_index], data[test_index]
	y_train, y_test = label[train_index], label[test_index]
	
	# choose best parameter
	# test linear kernel
	for d in deg:
		clf = SVC(kernel='poly', degree=d)
		svm_set = 'poly, degree=' + str(d)
		print 'train ' + svm_set
		print len(X_train), len(y_train)
		clf.fit(X_train, y_train)
		print 'after train do prediction'
		pred = clf.predict(X_test)
		print 'after prediction'

		roc = auc_score(y_test, pred)
		acc = accuracy_score(y_test, pred)
		apr = precision_score(y_test, pred)
		f1 = f1_score(y_test, pred)
	
		total = roc + acc + apr + f1
		print '5-fold CV No.', count_cv, ', setting: ', svm_set, 'total_score=', total

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
	print '5-fold CV No.', count_cv, ', setting: ', svm_set,', total_score=', total

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
		print '5-fold CV No.', count_cv, ', setting: ', svm_set, ', total_score=', total

		if total > max_score:
			max_score = total
			best_clf = clf
			best_roc = roc
			best_acc = acc
			best_apr = apr
			best_f1 = f1
			best_set = svm_set

	print '5-fold CV No.', count_cv, ', best score(CV) = ', (max_score / 4), ', setting: ', svm_set, 'best acc, roc, apr, f1(CV)=', best_acc, best_roc, best_apr, best_f1
	
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
# # try test set
# nf = [1,2,4,6,8,12,16,20]
# t_auc = 0
# t_acc = 0
# t_apr = 0
# t_f1 = 0

# for i in range(len(nf)):
# 	print nf[i]
# 	clf = RandomForestClassifier(n_estimators=1024, max_features=nf[i])
# 	clf.fit(data[0:5000], label[0:5000])
# 	#print clf.score(data[4500:5000], label[4500:5000])

# 	y = clf.predict(data[5000:20000])
# 	gt = label[5000:20000]

# 	auc = auc_score(gt, y)
# 	acc = accuracy_score(gt, y)
# 	apr = average_precision_score(gt, y)
# 	f1 = f1_score(gt, y)

# 	t_auc = t_auc + auc
# 	t_acc = t_acc + acc
# 	t_apr = t_apr + apr
# 	t_f1 = t_f1 + f1

# 	print "acc = ", acc
# 	print "roc = ", auc
# 	print "apr = ", apr
# 	print "f1  = ", f1

# print t_acc / 7
# print t_auc / 7
# print t_apr / 7
# print t_f1 / 7

#####################################################
# letter p1, train:test 4000:1000, feature=10, acc=92.9%
# letter p1, train:test 4000:1000, feature=10, acc=98.7%
# letter p1, train:test 4000:1000, feature=1, acc=98.2%

# letter.p1, 10-fold CV, feature=10, 0.9376
# letter.p2, 10-fold CV, feature=10, 0.987

#####################################################
# # doing cross validation
# kf = cross_validation.KFold(5000, n_folds=10)
# print kf

# total = 0
# count = 0
# for train_index, test_index in kf:
# 	# print("TRAIN:", train_index, "TEST:", test_index)
# 	X_train, X_test = data[train_index], data[test_index]
# 	y_train, y_test = label[train_index], label[test_index]
# 	clf.fit(X_train, y_train)
# 	total = total + clf.score(X_test, y_test)
# 	count = count + 1
# 	print total, count

# avg = total / count
# print total, count, avg