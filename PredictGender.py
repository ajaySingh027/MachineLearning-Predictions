###########################################################
##   Author : Ajay Singh                               ####
#            -----------@gmail.com                      ### 
##      Creating predictive fields from given data     #### 
########################################################### 
###########################################################  



from sklearn import tree, neighbors, neural_network

clf = tree.DecisionTreeClassifier()
clf_nehb = neighbors.KNeighborsClassifier()
clf_netwrk = neural_network.MLPClassifier()

# CHALLENGE - create 3 more classifiers...
# 1
# 2
# 3

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# CHALLENGE - ...and train them on our data
clf = clf.fit(X, Y)
clf_nehb = clf_nehb.fit(X, Y)
clf_netwrk = clf_netwrk.fit(X, Y)

prediction = clf.predict([[190, 70, 43]])
predi_2 = clf.predict([[185, 75, 39]])

pred_Ne_1 = clf_nehb.predict([[170, 68, 39]])
pred_netwrk = clf_netwrk.predict([[185, 75, 39]])

# CHALLENGE compare their reusults and print the best one!

print(prediction)
print(predi_2)
print("neighbor data ---------")
print(pred_Ne_1)
print("MLP classifies ------")
print(pred_netwrk)
