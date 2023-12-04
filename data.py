import numpy as np
import pandas as pd
import re
import random2
from math import ceil
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


class Dataset:
    """

    ANNOTATIONS FOR DATA:
    - English proficiency:
        1 - native speaker, 2 - advanced, 3 - moderate
    - Usual nightly sleep hours
    - How tired/excited/motivated a participant feels:
        1 - very much, 2 - somewhat, 3 - not at all
    - Scores of depression, anxiety and Big 5 personality traits
    - How often a participant reads news:
        1 - every day, 2 - more than half the days, 3 - a few days a week or less
    - How much time a participant spends reading news per week:
        1 - less than 5, 2 - 5-10h, 3 - more than 10h
    COL_NAMS:
     [1] "article_id"           "user_id"              "correctness"          "likability"
     [5] "usefulness"           "accuracy"             "confidence"           "change_view"
     [9] "clarity"              "total_points"         "EngProf"              "SleepHours"
    [13] "Tired"                "Excited"              "Motivated"            "Depression"
    [17] "Anxiety"              "Extraversion"         "Agreeableness"        "Conscientiousness"
    [21] "Neuroticism"          "OpennessToExperience" "HowOftenNews"         "TimeWeekNews"
    [25] "number_of_article"
    """

    def __init__(self, df: pd.DataFrame, data_type="whole", normalized=False):
        """

        :param df: original data
        :param data_type: character, "train", "validate", "test", default: "whole"
        :param normalized: Bool, default: False
        """
        self.row_usrId = None
        self.row_artId = None
        self.data = df
        # sub-dataset
        self.training = None
        self.validation = None
        self.testing = None
        # indexes
        self.read_count = None
        self.usrId = None
        self.artId = None
        self.allCol = (["article_id", "user_id", "correctness", "likability",
                        "usefulness", "accuracy", "confidence", "change_view",
                        "clarity", "total_points", "EngProf", "SleepHours",
                        "Tired", "Excited", "Motivated", "Depression", "Anxiety",
                        "Extraversion", "Agreeableness", "Conscientiousness",
                        "Neuroticism", "OpennessToExperience", "HowOftenNews",
                        "TimeWeekNews", "number_of_article"])
        self.inCol = (["EngProf", "SleepHours", "Tired", "Excited",
                       "Motivated", "Depression", "Anxiety", "Extraversion",
                       "Agreeableness", "Conscientiousness", "Neuroticism",
                       "OpennessToExperience", "HowOftenNews", "TimeWeekNews"])
        self.inColNumeric = (["SleepHours", "Depression", "Anxiety", "Extraversion",
                              "Agreeableness", "Conscientiousness",
                              "Neuroticism", "OpennessToExperience"])
        self.inColCat = (["EngProf", "Tired", "Excited", "Motivated", "HowOftenNews", "TimeWeekNews"])
        self.outCol = (["correctness", "likability", "usefulness",
                        "accuracy", "confidence", "change_view", "clarity"])
        # data_type: training, validation, testing, or whole
        # state: normalized True or False
        self.data_type = data_type
        self.state = normalized
        self._drop_na()
        self.row_zip_info = list(zip(self.data["article_id"],
                                     self.data["user_id"],
                                     self.data["total_points"],
                                     self.data["number_of_article"]))
        if not self.state:
            self.normalize()
        if self.data_type == "whole":
            self.fill_missing_data()
            self.get_training_data()

    def _drop_na(self):
        if self.data.isna().sum().sum() == 0:
            self.row_artId = self.data["article_id"] if "article_id" in self.data.columns else self.row_artId
            self.read_count = self.data["number_of_article"].unique() if "number_of_article" in self.data.columns \
                else self.read_count
            self.usrId = self.data["user_id"].unique() if "user_id" in self.data.columns else self.usrId
            self.artId = self.data["article_id"].unique() if "article_id" in self.data.columns else self.artId
            self.row_usrId = self.data["user_id"] if "user_id" in self.data.columns else self.row_usrId
        else:
            completed_rows = self.data.isna().sum(axis=1) == 0
            self.data = self.data.loc[completed_rows, :]
            self.data.loc[:, "number_of_article"] = self.data.groupby("user_id")["user_id"].transform("count")
            self.row_artId = self.data["article_id"]
            self.row_usrId = self.data["user_id"]
            self.read_count = self.data["number_of_article"].unique()
            self.usrId = self.data["user_id"].unique()
            self.artId = self.data["article_id"].unique()

    def normalize(self):
        # Preprocessing
        for i in ["EngProf", "Tired", "Excited", "Motivated", "HowOftenNews"]:
            self.data.loc[:, i] = self.data.loc[:, i] * (-1) + 2
        self.data.loc[:, "TimeWeekNews"] = self.data.loc[:, "TimeWeekNews"] - 2
        # normalization
        transformer = ColumnTransformer([
            ("num", StandardScaler(), self.outCol + self.inColNumeric),
            ("cat", OneHotEncoder(categories=[[-1, 0, 1]] * len(self.inColCat)), self.inColCat)
        ], remainder="passthrough")
        self.data = transformer.fit_transform(self.data)
        self.allCol = transformer.get_feature_names_out()
        self.allCol[-4:] = ["article_id", "user_id", "total_points", "number_of_article"]
        self.inCol = self.update_feature(self.inCol)
        self.inColCat = self.update_feature(self.inColCat)
        self.inColNumeric = self.update_feature(self.inColNumeric)
        self.outCol = self.update_feature(self.outCol)
        self.data = pd.DataFrame(self.data, columns=self.allCol)

    def update_feature(self, before):
        updated = []
        for i, feature in enumerate(before):
            regex = re.compile(f".*{re.escape(feature)}.*")
            updated = updated + list(filter(lambda x: re.match(regex, x) is not None, self.allCol))

        return np.asarray(updated)

    def fill_missing_data(self):
        """

        filling the article data where usr did not read with 0 in the whole dataset
        :return: a filled data
        """
        new_data = pd.DataFrame(columns=self.allCol)
        # append the data at the end of each usr
        # sort them finally
        for i, usr in enumerate(self.usrId):
            # subset rows with current usr ID
            row_select_bool = self.data["user_id"] == usr
            rows = self.data.loc[row_select_bool]
            total_points = list(set(rows["total_points"]))
            new_data = rows if new_data.empty else pd.concat([new_data, rows])
            # articles read by current usr
            read_art = self.data.loc[row_select_bool]["article_id"]
            # get articles not read by the current usr
            not_read = self.artId.tolist()
            for r in read_art:
                not_read.remove(r)
            # get a new row
            profile = rows[self.inCol].iloc[[-1], :].reset_index(drop=True)
            mark = pd.DataFrame(np.zeros(len(self.outCol)).reshape(1, 7),
                                columns=self.outCol).reset_index(drop=True)
            new_row = pd.concat([profile, mark], axis=1)
            repeat_rows = pd.DataFrame(np.repeat(new_row.values, len(not_read), axis=0),
                                       columns=new_row.columns).reset_index(drop=True)
            # adding article_id, user_id, total_points
            id_4_header = pd.DataFrame({"article_id": not_read,
                                        "user_id": np.repeat(usr, len(not_read)),
                                        "total_points": np.repeat(total_points, len(not_read)),
                                        "number_of_article": np.repeat(len(read_art), len(not_read))
                                        }).reset_index(drop=True)
            repeat_rows = pd.concat([id_4_header, repeat_rows], axis=1)
            # combine
            new_data = pd.concat([new_data, repeat_rows], axis=0)
        # sort
        self.data = new_data.sort_values(by=["user_id", "article_id"])

    def get_training_data(self):
        # Split the data into train and test
        train_id = random2.sample(self.usrId, ceil(0.7 * len(self.usrId)))
        train_data = self.data.loc[self.data["user_id"].isin(train_id)]
        test_id = self.usrId.tolist()
        for u in train_id:
            test_id.remove(u)
        test_data = self.data.loc[self.data["user_id"].isin(test_id)]
        train, test = train_test_split(self.data, test_size=0.3, random_state=42)
        # create train & testing objects
        self.training = Training(pd.DataFrame(train_data, columns=self.allCol), in_col=self.inCol, out_col=self.outCol)
        self.testing = Testing(pd.DataFrame(test_data, columns=self.allCol), in_col=self.inCol, out_col=self.outCol)


class BaseDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_type, normalized, in_col, out_col):
        super().__init__(df, data_type, normalized)
        self.inCol = in_col
        self.outCol = out_col
        self.input_matrix = self.data[self.inCol].to_numpy()
        self.out_matrix = self.data[self.outCol].to_numpy()
        self.transform_matrix()

    def transform_matrix(self):
        self.input_matrix = np.unique(self.input_matrix, axis=0)
        self.out_matrix = self.out_matrix.reshape(len(self.usrId),
                                                  len(self.artId),
                                                  len(self.outCol))
        self.out_matrix = self.out_matrix.transpose(2, 0, 1)


class Training(BaseDataset):
    def __init__(self, df: pd.DataFrame, normalized=True, in_col=None, out_col=None):
        super().__init__(df, data_type="train", normalized=normalized, in_col=in_col, out_col=out_col)


class Testing(BaseDataset):
    def __init__(self, df: pd.DataFrame, normalized=True, in_col=None, out_col=None):
        super().__init__(df, data_type="test", normalized=normalized, in_col=in_col, out_col=out_col)
