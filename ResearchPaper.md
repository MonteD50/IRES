Reasearch Paper


TODO:
- cite clustering based paper, kmodes, silhouette score 

Datasets to choose: 1, 4, 5, 6, 7, 12, 13, 16, 17 and 4 more from Kaggle

Abstarct: Get insipried by
    - Distance based Clustering of Class Association Rules to 
Build a Compact, Accurate and Descriptive Classifier paper


Intro: Get insipried by
    - Distance based Clustering of Class Association Rules to 
Build a Compact, Accurate and Descriptive Classifier paper


Related Work:
    - partially: Distance based Clustering of Class Association Rules to 
Build a Compact, Accurate and Descriptive Classifier paper
    - SPAR
    - CMAR
    - MAC


Methodology:
    - CAR generation: Distance based Clustering of Class Association Rules to 
Build a Compact, Accurate and Descriptive Classifier paper
        - support, confidence, etc.
        - fpgrowth and why we used: because faster/ better time complexity (find out what time complexity this and apriori is) and similiar results as apriori
    - Clustering: 
        - Distance: Explain and show example binary distance
            - Rephrase section 4.2 in PhD thesis
            - Hamming distance
        - Kmodes: because it is for categorical data
            - Identifying Optimal number of clusters: Shiloute score, how it works with algorithm
            - Explain kmodes
            - Show algorithm for our method: distance, optimal clusters, kmodes


Cluster Representation (rephrase):
    - 6.2: Representative CAR based on dataset coverage in PhD thesis
    - Algorithm (similiar to algorithm 14) but change to show avoid overlapping and three rules per cluster
    - Final classifier: combine all the steps, show as algorithm


Results:
    - Tables
        - Table 1: dataset info (See table 7.1 in PhD thesis without min support and min confidence and # of analyzed rules)
        - Table 2: Accuracy for each dataset (big table)
        - Table 3: Number of rules for each dataset (LATER: statistical significance testing)


Conclusion:
    - Rephrase from Distance based Clustering of Class Association Rules to 
Build a Compact, Accurate and Descriptive Classifier paper


Appendix:
    - graph of elbow and shilhoute score for nursary, mushroom, adult, connect4, and abolone datasets
    - maybe PCA plot of kmodes clustering


Chapter 4.1: Indirect ...
    Conditonal market basket probability distance



kmodes is partitional clustering

7/10/2023 Meeting Notes:
- TODO:
    - get 1 rule from closest to cluster center and 2 rules from coverage
    - DONE: abstract, intro/related work, Section 3.2
    - ~: run some expirements and find 4 datasets from Kaggle


- Title: A compact associative classification model using K-modes clustering with rule representations by coverage. 
- Abstract:
    - Generate cars using fpgrowth
    - Clustered using kmodes
        - Using distance metric of binarizing the rules
        - Find optimal clusters using silhouette score
    - Took representative rules based on coverage
    - Expiremental evaluation on UCI + Kaggle datasets
        - Acheived goal of reducing rule space dramatically


- Intro: Mostly higher level overview of the following but more in detail than abstract
    - Overall abt assocation rule mining
    - COmbine related works with intro
        - cite and introduce hierarchal clustering paper
    - Little abt cars-fpgrowth (advantages of fpgrowth vs apriori)
    - Conversion to binary dataset
    - why selected kmodes (better for categorical)
        - Optimal number of clusters
    - Coverage
    - Results


- Methadology:
    - Section 3.1, 3.3: reprhase once Jamol writes it
    - Section 3.2 clustering and silhouette score explanation in detail
    - Selected top 3 rules: 2 based on coverage because expiremental evaluation acheived highest coverage with 2 rules and 1 with closest to cluster center


- LATER:
    - run SPAR, CMAR, MAC and put in results


7/11/2023 Meeting Notes:
- Intro:
    - Right in more detail about the entire steps of the models. Jamol will write this

- Methadology:
    - Add a verbal description of kmodes algorithm
    - Algorithms:
        - DONE: Kmodes
        - DONE: Optimal # of clustering
        - Final model

TODO:
    - Instead of FURIA replace with SPAR and add CMAR, MAC in related works
    - Run on datasets + 4 Kaggle datasets
    - Add precision, recall, f1 score, and accuracy/std to results


7/13/2023 Meeting Notes:
TODO:
    - fix references + add references + rephrasre related work
    - Run on all datasets + 4 Kaggle
    - Run SPAR, CMAR, CBA
    - Add SPAR, CMAR, and CBA (is in literature review in PhD thesis) to related works
    - Add more to intro (one more paragraph on explaining the model in general)
    - Read + edit distance metric part
    - For table of results: Remove fr, pt and rdr and SPAR, CMAR, and our method
    - Rephrase final associative rule classifier
    - Remove tightness in related works

    Jonathan:
        - Add SPAR, CMAR, and CBA (is in literature review in PhD thesis) to related works
        - Add more to intro (one more paragraph on explaining the model in general)
        - Remove tightness in related works

    Charlie:
        - fix references + add references + rephrasre related work
        - find three more datasets

    Monte:
        - Run on all datasets + 4 Kaggle (need to do for 4 Kaggle, connect4, car evn, chess)
        - DONE: Run SPAR, CMAR, CBA (except for connect4)
        - DONE: Read + edit distance metric part

    After:
        - For table of results: Remove fr, pt and rdr and add SPAR, CMAR, and our method
        - Rephrase final associative rule classifier



Meeting Notes 7/13/2023:
    - TODO:
        - 2 Kaggle datasets + (failed)try 2 default datasets run
            - DONE: rerun mushroom on non-cheating
        - DONE: Add results to Table 2, 3
        - DONE: in related works in "polite way" say to the best of our knowledge since not much related work to the combination of class assocation rules and clustering. One paper used new distance metric wtih clustering on cars but not used for classification which is what our work does. Make it obvious the difference in that other paper used association rules not for classification
        - find another paper focused on cars and clustering or just cars for classification (after 2019)
        - DONE: Get number of rules for SPAR, CMAR, CBA
        - DONE: enlarge table and size of text of distance metric
        - graph of precision, recall, f1 score per model
        - DONE: graph of number of rules per model
        - DONE: bar graph of average accuracy and number of rules side by side per model
        - line graph of size of dataset vs number of rules foreach model
            - for datasets: chess, abalone, mushroom, nursery, adult, connect4 
            - for models: SA, CBA, CPAR, CMAR, our method

    - After taste:
        - Try to find 2 more datasets and run
        - find another paper focused on cars and clustering or just cars for classification (after 2019) and add to related works
        

Meeting Notes 7/17/2023:
- TODO:
    - GRAPH: Number of rules graph for airline reviews, airplane, connect4, adult
    - DONE: Run airplane, airline reviews again + 2 more datasets
    - GRAPH: bar graph of average accuracy and number of rules side by side per model

    - Add one sentence to abstract about Kaggle datasets
    - DONE: Add chess and connect4 to table


Meeting Notes 7/28/2023:
    - DONE: Number of rules graph for airline reviews, airplane, connect4, adult
    - DONE: bar graph of average accuracy and number of rules side by side per model
    - DONE: Same num rules graph but per dataset model (opposite)
    - DONE: Change names to ACMMode
    - DONE: line graph of size of dataset vs number of rules foreach model
            - for datasets: airline reviews, airplane, connect4, adult, abalone, mushroom, nursery, 
            - for models: SA, CBA, CPAR, CMAR, our method
            - NOT GOING TO DO BC UGLLYYY
    - DONE: Precision, recall, f1 measure averages of datasets

    - Put on github non- altered code
    - DONE: Rephrase all
    - DONE: One paragraph for figures: 
        - our model not sensitive to dataset size while others are
        - our model produced 10x better on average and better on cba and spar and better on other models 
        - DONE: on specific datasets performed similiar but with less rules
    - DONE: Paragrpah on better reuslts on specifc datasets and some better rules and sometimes couldnt
    - Add one sentence to abstract about each Kaggle datasets
    