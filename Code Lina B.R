############################################################################################
########################## Creation de partition aléatoire #################################
############################################################################################

# Importation des packages et libraries nécessaires
install.packages("ggplot2")
install.packages("C50")
install.packages("ROCR")
install.packages("randomForest")
install.packages("kknn")
install.packages("e1071")
install.packages("naivebayes")
library("e1071", lib.loc="~/R/win-library/3.6")
library("ROCR", lib.loc="~/R/win-library/3.6")
library("rpart", lib.loc="C:/Program Files/R/R-3.6.2/library")
library("rpart.plot", lib.loc="~/R/win-library/3.6")
library("caret", lib.loc="~/R/win-library/3.5")
library("C50", lib.loc="~/R/win-library/3.5")
library("tree", lib.loc="~/R/win-library/3.6")
library("randomForest", lib.loc="~/R/win-library/3.6")
library("kknn", lib.loc="~/R/win-library/3.6")


# Importation des données
Data <- read.csv("Data Projet.csv", header = TRUE, sep = ",", dec = ".")
#La variable customer n'est pas pertinente dans l'étude, on la retire
Data<-Data[,-3]
#La variable branch est de type int, nous devons la changer en variable catégorielle
Data$branch<-as.factor(Data$branch)
#La variable ed est de type catégorielle ordinale, on peut indiquer l'ordre des préférences
Data$ed<-factor(Data$ed, ordered = TRUE, levels= c("Niveau bac","Bac+2","Bac+3","Bac+4","Bac+5 et plus"))
# On ajoute une colonne index pour facilité la création de partition aléatoire
Data_with_index <- data.frame(id=1:length(Data$default),Data)
#set.seed pour rendre reproductible les résultats
set.seed(100)
# Index pour l'ensemble d'apprentissage (70%)
index_EA<-createDataPartition(Data_with_index$default, times = 1, p = 0.7,list = FALSE)
# On selectionne uniquement les ligne dont id est dans index_EA
EA_with_index <- Data_with_index[index_EA,]
# On supprime la colonne id
EA<-EA_with_index[-1]
# Idem pour l'ensemble de test en excluant les ligne dont l'index est dans index_EA:
ET_with_index <- Data_with_index[-index_EA,]
# Idem, On supprime la colonne id
ET<-ET_with_index[-1]

# Verification du split:
nrow(EA)/nrow(Data)  # 70% ok
nrow(ET)/nrow(Data)  # 30% ok
print("Data")
print(prop.table(table(Data$default))) #37,3% de 'Oui' dans Data
print("EA")
print(prop.table(table(EA$default))) #37,3% de 'Oui' dans EA
print("ET")
print(prop.table(table(ET$default))) #37,3% de 'Oui' dans ET
#Le split semble être bien réalisé

#default= Oui (positif) -> le client ne rembourse pas
#default=Non (négatif)  _> le client rembourse
#Ce qui nous intéresse dans chacun des classifieurs, c'est de minimiser
#le taux de FAUX NEGATIF, c'est à dire minimiser les erreures de prédictions
#NEGATIVE (alors qu'elles sont en réalité POSITIVE).
#De cette manière, on minimise le risque de défaut de paiement.
#Nous allons donc : -Minimiser le taux de faux négatif (Défaut de paiement prédit remboursé)
#                   -Maximiser le taux de vrai négatif (Remboursement de l'emprunt prédit remboursé)


######################################################################################################

############## Testons un premier classifieur rpart ##################################################


#On stock l'arbre construit par l'ensemble d'apprentissage dans Tree1
Tree1 <- rpart(default ~ .,EA)
#On affiche notre arbre (avec prp qui permet une meilleure vision de l'arbre)
prp(Tree1)
#En indiquant le texte correspondant à chaque branche
text(Tree1, pretty = 0)

#Calcul du taux de succès: 
#Application de Tree1 à ET 
test_Tree1 <- predict(Tree1, ET, type="class")
print(test_Tree1)
table(test_Tree1)

# Stockage des résultats dans un data frame df_Tree1 
df_Tree1 <- as.data.frame(table(ET$default, test_Tree1)) 

# Renommage des colonnes dans le data frame df_tree1 
colnames(df_Tree1) = list("Classe", "Prediction", "Effectif") 

# Calcul de la proportion de succès parmi le nombre total d'exemples de test 
sum(df_Tree1[df_Tree1$Classe==df_Tree1$Prediction,"Effectif"])/nrow(ET)

#TAUX DE SUCCES : [1] 0.7214485

#COURBE ROC ET INDICE AUC
prob_Tree1 <- predict(Tree1,ET, type="prob")
roc_pred_Tree1 <- prediction(prob_Tree1[,2],ET$default) 
roc_perf_Tree1 <- performance(roc_pred_Tree1,"tnr","fnr")
plot(roc_perf_Tree1, col = "green") 
auc_Tree1 <- performance(roc_pred_Tree1, "auc")
attr(auc_Tree1, "y.values")

#INDICE AUC DU 1ER CLASSIFIEUR
#[1] 0.7432504

par(new=TRUE)
#########################################################################################

############## Second classifieur C5.0 ##################################################

#On stock l'arbre construit par l'ensemble d'apprentissage dans Tree2
Tree2 <- C5.0(default ~ ., EA)
#On affiche notre arbre
#plot(Tree2)

#Calcul du taux de succès: 
#Application de Tree2 à ET 
test_Tree2 <- predict(Tree2, ET, type="class")
print(test_Tree2)
table(test_Tree2)

# Stockage des résultats dans un data frame df_Tree2
df_Tree2 <- as.data.frame(table(ET$default, test_Tree2)) 

# Renommage des colonnes dans le data frame df_tree2 
colnames(df_Tree2) = list("Classe", "Prediction", "Effectif") 

# Calcul de la proportion de succès parmi le nombre total d'exemples de test 
sum(df_Tree2[df_Tree2$Classe==df_Tree2$Prediction,"Effectif"])/nrow(ET)

#TAUX DE SUCCES : [1] 0.7130919

#COURBE ROC ET INDICE AUC
prob_Tree2 <- predict(Tree2,ET, type="prob")
roc_pred_Tree2 <- prediction(prob_Tree2[,2],ET$default) 
roc_perf_Tree2 <- performance(roc_pred_Tree2,"tnr","fnr")
plot(roc_perf_Tree2, col = "blue") 
auc_Tree2 <- performance(roc_pred_Tree2, "auc")
attr(auc_Tree2, "y.values")

#INDICE AUC DU 2nd CLASSIFIEUR
#[1] 0.7174295

par(new=TRUE)
#########################################################################################

############## Troisième classifieur tree ##################################################


#On stock l'arbre construit par l'ensemble d'apprentissage dans Tree3
Tree3<-tree(default ~ ., EA)
#On affiche notre arbre
plot(Tree3)
#En indiquant le texte correspondant à chaque branche
text(Tree3, pretty = 0)


#Calcul du taux de succès: 
#Application de Tree3 à ET 
test_Tree3 <- predict(Tree3, ET, type="class")
print(test_Tree3)
table(test_Tree3)

# Stockage des résultats dans un data frame df_Tree3 
df_Tree3 <- as.data.frame(table(ET$default, test_Tree3)) 

# Renommage des colonnes dans le data frame df_Tree3 
colnames(df_Tree3) = list("Classe", "Prediction", "Effectif") 

# Calcul de la proportion de succès parmi le nombre total d'exemples de test 
sum(df_Tree3[df_Tree3$Classe==df_Tree3$Prediction,"Effectif"])/nrow(ET)

#TAUX DE SUCCES : [1] 0.6852368


#COURBE ROC ET INDICE AUC
prob_Tree3 <- predict(Tree3,ET, type="vector")
roc_pred_Tree3 <- prediction(prob_Tree3[,2],ET$default) 
roc_perf_Tree3 <- performance(roc_pred_Tree3,"tnr","fnr")
plot(roc_perf_Tree3, col = "red") 
auc_Tree3 <- performance(roc_pred_Tree3, "auc")

#INDICE AUC DU 3eme CLASSIFIEUR
attr(auc_Tree3, "y.values")

#INDICE AUC DU 3eme CLASSIFIEUR
#[1] 0.7157711

par(new=TRUE)


#########################################################################################

############## Quatrième classifieur randomForest  ##################################################


#On stock l'arbre construit par l'ensemble d'apprentissage dans Tree4
Tree4<-randomForest(default ~ ., EA)
#On affiche notre arbre, ici ce n'est pas un arbre mais un graphique représentant
#les taux d'erreure de prediction de la classe Non, Oui, et totale des deux
plot(Tree4)


#Calcul du taux de succès: 
#Application de Tree4 à ET 
test_Tree4 <- predict(Tree4, ET, type="class")
print(test_Tree4)
table(test_Tree4)

# Stockage des résultats dans un data frame df_Tree4
df_Tree4 <- as.data.frame(table(ET$default, test_Tree4)) 

# Renommage des colonnes dans le data frame df_tree4
colnames(df_Tree4) = list("Classe", "Prediction", "Effectif") 

# Calcul de la proportion de succès parmi le nombre total d'exemples de test 
sum(df_Tree4[df_Tree4$Classe==df_Tree3$Prediction,"Effectif"])/nrow(ET)

#TAUX DE SUCCES : [1] 0.724234


#COURBE ROC ET INDICE AUC
prob_Tree4 <- predict(Tree4,ET, type="prob")
roc_pred_Tree4 <- prediction(prob_Tree4[,2],ET$default) 
roc_perf_Tree4 <- performance(roc_pred_Tree4,"tnr","fnr")
plot(roc_perf_Tree4, col = "black") 
auc_Tree4 <- performance(roc_pred_Tree4, "auc")
attr(auc_Tree4, "y.values")

#INDICE AUC DU 4eme CLASSIFIEUR
#[1] 0.7821393

par(new=TRUE)

#########################################################################################

############## Cinquième classifieur kknn  ##################################################


Tree5 <- function(arg1, arg2, arg3, arg4){
  # Apprentissage et test simultanes du classifeur de type k-nearest neighbors 
  kknn <- kknn(default~.,EA,ET, k = arg1, distance = arg2)
  # Matrice de confusion 
  matConfusion<-table(ET$default, kknn$fitted.values) #fitted values désigne les classes
  print(matConfusion)
  # Conversion des probabilites en data frame 
  kknn_prob <- as.data.frame(kknn$prob)
  # Courbe ROC 
  kknn_pred <- prediction(kknn_prob$Oui, ET$default) 
  kknn_perf <- performance(kknn_pred,"tnr","fnr")
  plot(kknn_perf, main = "Classifeurs K-plus-proches-voisins kknn()", add = arg3, col = arg4)
  # Calcul de l'AUC et affichage par la fonction cat() 
  kknn_auc <- performance(kknn_pred, "auc") 
  cat("AUC = ", as.character(attr(kknn_auc, "y.values")))
  # Return sans affichage sur la console 
  invisible()
}

Tree5_1<-Tree5(10,1,TRUE,'orange')
Tree5_2<-Tree5(10,2,TRUE,'magenta')
Tree5_3<-Tree5(20,1,TRUE,'purple')
Tree5_4<-Tree5(20,2,TRUE,'dark green')
Tree5_4<-Tree5(20,2,TRUE,'orange')
par(new=TRUE)

#>Tree5(10,1,TRUE,'orange')
#     Non Oui
#Non 171  54
#Oui  71  63
#AUC =  0.711243781094528

#> Tree5(10,2,TRUE,'pink')
#     Non Oui
#Non 171  54
#Oui  71  63
#AUC =  0.685638474295191

#> Tree5(20,1,TRUE,'light blue')
#     Non Oui
#Non 171  54
#Oui  70  64
#AUC =  0.732139303482587

#> Tree5(20,2,TRUE,'dark green')
#     Non Oui
#Non 171  54
#Oui  67  67
#AUC =  0.711674958540631

#########################################################################################

############## Sixième classifieur svm  ##################################################


#Pour le classifieur SVM
#Definition de la fonction d'apprentissage, Tree6 et evaluation 
Tree6 <- function(arg1, arg2, arg3){ 
  # Apprentissage du classifeur 
  svm <- svm(default~., EA, probability=TRUE, kernel = arg1)
  # Test du classifeur : classe predite 
  svm_class <- predict(svm, ET, type="response")
  # Matrice de confusion 
  print(table(ET$default, svm_class))
  # Test du classifeur : probabilites pour chaque prediction 
  svm_prob <- predict(svm, ET, probability=TRUE)
  # Recuperation des probabilites associees aux predictions 
  svm_prob <- attr(svm_prob, "probabilities")
  # Courbe ROC 
  svm_pred <- prediction(svm_prob[,2], ET$default)
  svm_perf <- performance(svm_pred,"tnr","fnr") 
  plot(svm_perf, main = "Support vector machines svm()", add = arg2, col = arg3)
  # Calcul de l'AUC et affichage par la fonction cat() 
  svm_auc <- performance(svm_pred, "auc") 
  cat("AUC = ", as.character(attr(svm_auc, "y.values")))
  # Return sans affichage sur la console 
  invisible()
}

Tree6("linear", TRUE, "red") 
Tree6("polynomial", TRUE, "blue")
Tree6("radial", TRUE, "green") 
Tree6("sigmoid", TRUE, "orange")

#Tree6("linear", TRUE, "red") 
#svm_class
#    Non Oui
#Non 186  39
#Oui  59  75
#AUC =  0.793399668325042

#> Tree6("polynomial", TRUE, "blue")
#svm_class
#    Non Oui
#Non 223   2
#Oui 128   6
#AUC =  0.769585406301824

#> Tree6("radial", TRUE, "green") 
#svm_class
#   Non Oui
#Non 191  34
#Oui  64  70
#AUC =  0.783449419568824

#> Tree6("sigmoid", TRUE, "orange")
#svm_class
#    Non Oui
#Non 185  40
#Oui  60  74
#AUC =  0.779701492537313


#Pour le classifieur naivebayes
# Definition de la fonction d'apprentissage, test et evaluation
Tree7 <- function(arg1, arg2, arg3, arg4){
  # Apprentissage du classifeur 
  nb <- naive_bayes(default~., EA, laplace = arg1, usekernel = arg2)
  # Test du classifeur : classe predite 
  nb_class <- predict(nb, ET, type="class")
  # Matrice de confusion 
  print(table(ET$default, nb_class))
  # Test du classifeur : probabilites pour chaque prediction 
  nb_prob <- predict(nb, ET, type="prob")
  # Courbe ROC 
  nb_pred <- prediction(nb_prob[,2], ET$default) 
  nb_perf <- performance(nb_pred,"tnr","fnr") 
  plot(nb_perf, main = "Classifieurs bayésiens naïfs naiveBayes()", add = arg3, col = arg4)
  # Calcul de l'AUC et affichage par la fonction cat() 
  nb_auc <- performance(nb_pred, "auc") 
  cat("AUC = ", as.character(attr(nb_auc, "y.values")))
  # Return sans affichage sur la console 
  invisible()
}

# Naive Bayes 
Tree7_1<-Tree7(0, FALSE, FALSE, "black") 
Tree7_2<-Tree7(20, FALSE, TRUE, "blue") 
Tree7_3<-Tree7(0, TRUE, TRUE, "green") 
Tree7_4<-Tree7(20, TRUE, TRUE, "orange")


#> Tree7_4<-Tree7(20, TRUE, TRUE, "orange")
#nb_class
#     Non Oui
#Non 153  72
#Oui  46  88
#AUC =  0.744212271973465

#> Tree7_3<-Tree7(0, TRUE, TRUE, "green")
#nb_class
#     Non Oui
#Non 155  70
#Oui  50  84
#AUC =  0.741724709784411

#> Tree7_2<-Tree7(20, FALSE, TRUE, "blue")
#nb_class
#     Non Oui
#Non 147  78
#Oui  37  97
#AUC =  0.758573797678276

# Naive Bayes 
#> Tree7_1<-Tree7(0, FALSE, FALSE, "black")
#nb_class
#     Non Oui
#Non 149  76
#Oui  41  93
#AUC =  0.752968490878939

#######################################################################################
#######################################################################################

#Nouvelle base de données
Datanew<- read.csv("Data Projet New.csv", header = TRUE, sep = ",", dec = ".")
#Sur laquelle on effectue les mêmes modifications
Datanew2<-Datanew[,-3]
#La variable branch est de type int, nous devons la changer en variable catégorielle
Datanew2$branch<-as.factor(Datanew$branch)
#La variable ed est de type catégorielle ordinale, on peut indiquer l'ordre des préférences
Datanew2$ed<-factor(Datanew$ed, ordered = TRUE, levels= c("Niveau bac","Bac+2","Bac+3","Bac+4","Bac+5 et plus"))

#On applique notre classifieur à nos nouvelles données

svm <- svm(default~., EA, probability=TRUE, kernel = 'linear')
svm_class <- predict(svm, Datanew2, type="response")
print(svm_class)
default<-as.matrix(svm_class)
print(default)
resultats2<- as.data.frame(default) 

#Dans resultat2 se trouve toutes les prédictions de chaque clients
print(resultats2)

svm_prob <- predict(svm, Datanew2, probability=TRUE)
svm_prob <- attr(svm_prob, "probabilities")

#Dans svm_prob se trouvent les probabilités associées aux oui et non de chaque clients
print(svm_prob)

#On fait une boucle pour prendre les probabilités de la classe prédite uniquement
proba<-c()
resultat3<-as.matrix(resultats2)
j=1
for(i in resultat3){
  if(i=="Oui"){
    proba<-c(proba,svm_prob[,2][j])
    j=j+1
  }
  else{
    proba<-c(proba,svm_prob[,1][j])
    j=j+1
  }
  
}
print(proba) #Dans proba se trouve toutes les probabilités associées à chaque prédictions

#On stock les identifiants dans l'ordre
identifiant<- as.matrix(Datanew$customer)

#On stock les prédictions dans l'ordre
prédiction<-resultat3

#On stock les probabilités dans l'ordre
probabilité<-as.matrix(proba)

#On cré le tableau contenant les identifiants, prédictions et probabilités
Tableau<- matrix(ncol=3,nrow=300)
Tableau[,1]<-identifiant
Tableau[,2]<-prédiction
Tableau[,3]<-probabilité
colnames(Tableau)=list("Identifiant","Prédiction","Probabilité")

#On écrit nos résultats dans un ficher.csv
write.csv(Tableau,file='Résultats.csv')



