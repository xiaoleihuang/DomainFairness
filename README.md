# Adaptation Methods to Promote Multilingual Fairness NAACL - 2022
Data and code repository to promote multilingual fairness for hate speech detection and recommendation system (rating prediction) for the NAACL 2022 paper [Easy Adaptation to Mitigate Gender Bias in Multilingual Text Classification](https://arxiv.org/pdf/2204.05459.pdf).


# Data
  * I have released the data resources in this link: https://github.com/xiaoleihuang/DomainFairness/blob/main/data/data_list.txt

* Data Languages
    * Hate Speech:
      * English
      * Italian
      * Portugese
      * Spanish
    * Review (Recommendation System)
      * English
      * French
      * German
      * Danish


# How to Run
* Install the following:;
  * Install [conda](https://www.anaconda.com/distribution/);
  * Install Python packages:
    * With conda: `conda env create -f environment.yml`, then `conda activate naacl2022`
    * With pip: `pip install -r requirements.txt`.

* Download embeddings under the `resources/embeddings/`. Unzip the gz files and save the file under the folder.
* I have provided the multilingual lexicon resources, but you are free to run the codes under the `resources/lexicon/` to get your own version.
* Run Python scripts
  * Preprocessing: the hate speech data has been well formatted, while we have to process the review data. Luckily, I release the preprocessing codes for fully replicate. You can find the code in this link: https://github.com/xiaoleihuang/DomainFairness/blob/main/data/trustpilot_extractor.py
  * Data stats: After the preprocessing, you can get data staistics by this script: https://github.com/xiaoleihuang/DomainFairness/blob/main/data/data_stats.py
  * Baselines: `cd baseline/` to obtain performance of baselines;
  * Easy Adaptation Method: `cd model/` to obtain performance of my proposed approach;
    * The shell script provides examples of running my proposed method.


# Poster and Presentation
You can find my poster and presentation files under the `resources` directory.

# Contact
Please email **xiaolei.huang@memphis.edu** for further discussion.


# Citation
If you use our corpus in your publication, please kindly cite this [paper](https://arxiv.org/pdf/2204.05459.pdf):

```
@inproceedings{huang2022-naacl,
    title = "Easy Adaptation to Mitigate Gender Bias in Multilingual Text Classification",
    author = "Huang, Xiaolei",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = july,
    year = "2022",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/pdf/2204.05459.pdf",
    doi = "",
    pages = "",
    abstract = "Existing approaches to mitigate demographic biases evaluate on monolingual data, however, multilingual data has not been examined. In this work, we treat the gender as domains (e.g., male vs. female) and present a standard domain adaptation model to reduce the gender bias and improve performance of text classifiers under multilingual settings. We evaluate our approach on two text classification tasks, hate speech detection and rating prediction, and demonstrate the effectiveness of our approach with three fair-aware baselines.",
}
```
