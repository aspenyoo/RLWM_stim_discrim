
# load libraries
library(aod)

# #  # =============== 2 way rm ANOVA! =========

# load data
dir <- "/Users/aspen/Documents/Research/exemplar"
setwd(dir)
exptype = "RPP"
if (exptype == "RPP") {
  nSubjs <- 30
} else {
  nSubjs <-59
}

mydata <- read.csv(sprintf("dataforanova_%s.csv",exptype),header=TRUE)

mydata$SUBJECT <- as.factor(mydata$SUBJECT)
mydata$CONDITION <- as.factor(mydata$CONDITION)
mydata$SETSIZE <- as.factor(mydata$SETSIZE)

conds.aov <- with(mydata,
                   aov(PC ~ CONDITION * SETSIZE +
                         Error(SUBJECT / (CONDITION * SETSIZE))))
summary(conds.aov)
tapply(mydata$PC,mydata$CONDITION,mean)
tapply(mydata$PC,mydata$CONDITION,sd)/sqrt(nSubjs)
pairwise.t.test(mydata$PC, mydata$CONDITION, p.adj = "bonf")

# setsize conditioned anova
df1 <- mydata[mydata$SETSIZE=="3",]
df2 <- mydata[mydata$SETSIZE=="6",]

attach(df1)
tapply(PC,CONDITION,mean)
tapply(PC,CONDITION,sd)/sqrt(nSubjs)
summary(aov(PC ~ CONDITION + Error(SUBJECT / CONDITION)))
pairwise.t.test(PC, CONDITION, p.adj = "bonf")
detach()

attach(df2)
tapply(PC,CONDITION,mean)
tapply(PC,CONDITION,sd)/sqrt(nSubjs)
summary(aov(PC ~ CONDITION + Error(SUBJECT / CONDITION)))
pairwise.t.test(PC, CONDITION, p.adj = "bonf")
detach()

# #  TORTOISE AND HARE
mydata <- read.csv("dataforanova_tortoisehare.csv",header=TRUE)
mydata$subject <- as.factor(mydata$subject)
mydata$condition <- as.factor(mydata$condition)

conds.aov <- with(mydata,
                  aov(TH ~ condition +
                        Error(subject / (condition ))))
summary(conds.aov)
