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
    - FINISH: abstract, intro/related work, Section 3.2
    - FINISH: run some expirements and find 4 datasets from Kaggle


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