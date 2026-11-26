Tabular foundation models are especially interesting when every labeled sample is expensive. But in experimental science, 1,080 rows can still mean just 72 independent samples.

I used the Mice Protein Expression dataset to compare TabPFN v2 and NVIDIA Kumo Tabular Small with conventional classifiers. Each mouse has 15 repeated protein measurements, so I held out whole mice and then drew matched training subsets while keeping the test mice fixed.

With 16 labeled mice, Kumo and regularized logistic regression scored 0.435 and 0.358 in a matched exploratory learning curve. With 48, they scored 0.780 and 0.657. The average gap grew, but the direction varied across training-subset draws. Randomly splitting measurement rows, by contrast, made every tested model look nearly perfect.

My practical takeaway: count independent experimental units before calling a table “small,” and keep the held-out units fixed when asking whether a model learns more from fewer labels. This is one cohort, and CatBoost, XGBoost, and newer TabPFN checkpoints still need evaluation.

Full article: [LINK TO ARTICLE]

Where have you seen foundation models help most in scientific tables: at the first few labeled experiments, or once there is enough context to characterize the task?

#ScientificML #TabularML #MachineLearning

Accompanying figure: `results/figures/linkedin_learning.png`.

Optional mentions when posting: Prior Labs for TabPFN v2; NVIDIA for Kumo Tabular Small. Neither organization is an endorser of this study.
