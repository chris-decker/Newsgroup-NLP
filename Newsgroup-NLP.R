library(tm)
library(SnowballC)
library(class)
library(dplyr)
library(Rtsne)
library(caret)

#Function to create corpora from text files
corp.create <- function(type, subject) {
  Path.Loc <-
    system.file('texts',
                '20Newsgroups',
                paste0('20news-bydate-', type),
                subject,
                package = 'tm')
  Files <- DirSource(Path.Loc)
  return (VCorpus(
    URISource(Files$filelist[1:200]),
    readerControl = list(reader = readPlain)
  ))
}

#4 corpora for each of med/baseball and train/test
Doc1.Train <- corp.create('train', 'sci.med')
Doc1.Test <- corp.create('test', 'sci.med')
Doc2.Train <- corp.create('train', 'rec.sport.baseball')
Doc2.Test <- corp.create('test', 'rec.sport.baseball')

#Function to extract fields from corpora
get.fields <- function(field, doc) {
  field.vals <- c()
  for (i in seq(1:length(doc))) {
    field.vals <-
      c(field.vals, substr(doc[[i]]$content[grep(field, doc[[i]]$content)[1]], nchar(field) + 2, 1000000L))
  }
  return (field.vals)
}

#Subject extraction
Sub1.Train <- get.fields('Subject:', Doc1.Train)
Sub1.Test <- get.fields('Subject:', Doc1.Test)
Sub2.Train <- get.fields('Subject:', Doc2.Train)
Sub2.Test <- get.fields('Subject:', Doc2.Test)

#Email extraction
Eml1.Train <- get.fields('From:', Doc1.Train)
Eml1.Test <- get.fields('From:', Doc1.Test)
Eml2.Train <- get.fields('From:', Doc2.Train)
Eml2.Test <- get.fields('From:', Doc2.Test)

#Organization extraction
Org1.Train <- get.fields('Organization:', Doc1.Train)
Org1.Test <- get.fields('Organization:', Doc1.Test)
Org2.Train <- get.fields('Organization:', Doc2.Train)
Org2.Test <- get.fields('Organization:', Doc2.Test)

#Get unique email counts
emails <-
  data.frame(
    Sci.Tr.Eml = Eml1.Train,
    Sci.Ts.Eml = Eml1.Test,
    Bbl.Tr.Eml = Eml2.Train,
    Bbl.Ts.Eml = Eml2.Test
  )
eml.uniq <- emails %>%
  summarise_all(funs(n_distinct))
eml.uniq <-
  data.frame(Count = t(eml.uniq),
             Posts.Email = t(200 / eml.uniq))
eml.uniq

#Find most prolific poster & # of posts
eml.cnt <- table(orgemails$Email)
names(which.max(eml.cnt))
max(eml.cnt)

#Get sample of prolific poster's posts
emailsub <-
  data.frame(
    Email = c(Eml1.Train, Eml1.Test, Eml2.Train, Eml2.Test),
    Sub = c(Sub1.Train, Sub1.Test, Sub2.Train, Sub2.Test)
  )
head(unique(subset(emailsub, Email == names(which.max(
  eml.cnt
)))$Sub), 10)

#Create corpora from extracted subjects & combine
Sub1.Train.corp <- VCorpus(VectorSource(Sub1.Train))
Sub1.Test.corp <- VCorpus(VectorSource(Sub1.Test))
Sub2.Train.corp <- VCorpus(VectorSource(Sub2.Train))
Sub2.Test.corp <- VCorpus(VectorSource(Sub2.Test))
sub.corp <-
  c(Sub1.Train.corp,
    Sub1.Test.corp,
    Sub2.Train.corp,
    Sub2.Test.corp)

#Function to view the first 10 subjects
view_content <- function(corp) {
  for (i in seq(1:10)) {
    print(corp[[i]]$content)
  }
}

#Apply pre-processing and monitor changes made
view_content(sub.corp)
sub.tranf <-
  tm_map(sub.corp, removePunctuation)
view_content(sub.tranf)
sub.tranf <-
  tm_map(sub.tranf, content_transformer(tolower))
view_content(sub.tranf)
sub.tranf <-
  tm_map(sub.tranf, removeNumbers)
view_content(sub.tranf)
sub.tranf <-
  tm_map(sub.tranf, removeWords, stopwords('english'))
view_content(sub.tranf)
sub.tranf <-
  tm_map(sub.tranf, removeWords, 're')
view_content(sub.tranf)
sub.tranf <-
  tm_map(sub.tranf, stemDocument)
view_content(sub.tranf)

#Create Document-Term Matrix from the subject corpus
sub.dtm <-
  DocumentTermMatrix(sub.tranf, control = list(wordLengths = c(1, Inf),
                                               bounds = list(global = c(5, Inf))))
inspect(sub.dtm)

#View most frequent terms in corpus
freq <- colSums(as.matrix(sub.dtm))
ord <- order(freq, decreasing = T)
View(head(freq[ord], 10))

#Split DTM into test & train
sub.train <- as.matrix(sub.dtm)[c(1:200, 401:600), ]
sub.test <- as.matrix(sub.dtm)[c(201:400, 601:800), ]
sub.train %>% glimpse
sub.test %>% glimpse

#Create factor of correct tags to compare to
correct.tags <- factor(c(rep('Med', 200), rep('Bbl', 200)))
correct.tags

#Perform k-nearest neighbors classification and write results to dataframe
set.seed(123)
knn.out <- knn(sub.train,
               sub.test,
               correct.tags,
               k = 1,
               prob = TRUE)
knn.out %>% glimpse
a <- 1:length(knn.out)
b <- levels(knn.out)[knn.out]
c <- attributes(knn.out)$prob
d <- knn.out == correct.tags
result <- data.frame(
  Doc = a,
  Predict = b,
  Prob = c,
  Correct = d
)
result %>% glimpse

#Calculate total percentage of correct classifications
prop.table(table(result$Correct)) * 100

#Generate confusion matrix, precision, recall, and F-score for classification results
confusionMatrix(result$Predict, correct.tags, mode = 'prec_recall')

#Generate plot visualizing confusion matrix
qplot(
  correct.tags,
  result$Predict,
  colour = correct.tags,
  geom = c('jitter'),
  main = 'Confusion Matrix',
  xlab = 'Actual',
  ylab = 'Predicted'
) + scale_y_discrete(limits = rev(levels(result$Predict))) + scale_x_discrete(position = 'top') + theme(legend.position = 'none')

#Plot t-SNE reduction of test data
tsne <- Rtsne(sub.test, check_duplicates = F)
simp.tags = factor(c(rep('m', 200), rep('b', 200)))
plot(
  tsne$Y,
  t = 'n',
  main = 't-SNE Dimensionality Reduction',
  xlab = '',
  ylab = ''
)
text(tsne$Y, labels = simp.tags, col = c(rep('red', 200), rep('blue', 200)))
