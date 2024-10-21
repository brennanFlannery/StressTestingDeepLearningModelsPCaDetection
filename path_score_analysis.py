import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import scipy.stats as stats


bx_file = "E:/Brennan/bxf169/PathValidationAnalysis/BX Tiles Analyzed.xlsx"
rp_file = "E:/Brennan/bxf169/PathValidationAnalysis/RP Tiles Analyzed.xlsx"
subtypes = ["cribriform", "foamy", "intraductal", "inflammation", "atrophy", "necrosis", "ductal", "stroma"]
by_quad = False
by_grade = True
bx_data = pd.read_excel(bx_file)
rp_data = pd.read_excel(rp_file)

# Reorganize data frames to be usable for plotting
#
bx_data_mr = bx_data[["ground_truth", "Mr_prediction", "Mr_result", "gleason_grade_pattern_1",
                      "gleason_grade_pattern_2", "Subtype"]].copy()
bx_data_mr = bx_data_mr.rename(columns={"Mr_prediction":"prediction", "Mr_result": "result"})
bx_data_mr["Model"] = ["Mr" for x in range(len(bx_data_mr))]
bx_data_mr["Data"] = ["BX" for x in range(len(bx_data_mr))]
#
bx_data_mb = bx_data[["ground_truth", "Mb_prediction", "Mb_result", "gleason_grade_pattern_1",
                      "gleason_grade_pattern_2", "Subtype"]].copy()
bx_data_mb = bx_data_mb.rename(columns={"Mb_prediction":"prediction", "Mb_result": "result"})
bx_data_mb["Model"] = ["Mb" for x in range(len(bx_data_mb))]
bx_data_mb["Data"] = ["BX" for x in range(len(bx_data_mb))]
#
rp_data_mr = rp_data[["ground_truth", "Mr_prediction", "Mr_result", "gleason_grade_pattern_1",
                      "gleason_grade_pattern_2", "Subtype"]].copy()
rp_data_mr = rp_data_mr.rename(columns={"Mr_prediction":"prediction", "Mr_result": "result"})
rp_data_mr["Model"] = ["Mr" for x in range(len(rp_data_mr))]
rp_data_mr["Data"] = ["RP" for x in range(len(rp_data_mr))]
#
rp_data_mb = rp_data[["ground_truth", "Mb_prediction", "Mb_result", "gleason_grade_pattern_1",
                      "gleason_grade_pattern_2", "Subtype"]].copy()
rp_data_mb = rp_data_mb.rename(columns={"Mb_prediction":"prediction", "Mb_result": "result"})
rp_data_mb["Model"] = ["Mb" for x in range(len(rp_data_mb))]
rp_data_mb["Data"] = ["RP" for x in range(len(rp_data_mb))]

# Stack dataframes
master_data = pd.concat([rp_data_mb, rp_data_mr, bx_data_mb, bx_data_mr])
#


quad_dict = {0:"TN", 1:"FN", 2:"FP", 3:"TP"}

# Add data to
# Transform Mb and Mr results using quad dict
master_data['result'] = master_data['result'].apply(lambda x: quad_dict[x])


# Transform grade to a categorical variable
master_data['gleason_grade_pattern_1'] = master_data['gleason_grade_pattern_1'].apply(lambda x: "No Cancer" if x==0 else x)
master_data['gleason_grade_pattern_1'] = master_data['gleason_grade_pattern_1'].astype("category")
master_data['gleason_grade_pattern_2'] = master_data['gleason_grade_pattern_2'].astype("category")
master_data = master_data.reset_index()

if by_quad:
    subtype_dict = {"subtypes":subtypes}
    for quad in quad_dict.values():
        for datatype in master_data["Data"].unique():
            reduced_data = master_data[master_data["result"]==quad]
            reduced_data = reduced_data[reduced_data["Data"]==datatype]
            if quad in ["TN", "FP"]:
                reduced_data = reduced_data[reduced_data["gleason_grade_pattern_1"]=="No Cancer"]
            if quad in ["FN", "TP"]:
                reduced_data = reduced_data[reduced_data["gleason_grade_pattern_1"]!="No Cancer"]

            subtype_dict[f"{quad}_{datatype}_mb"] = [np.sum([(y in str(x).lower())*1 for x in reduced_data[reduced_data["Model"]=="Mb"]["Subtype"]]) for y in subtypes]
            subtype_dict[f"{quad}_{datatype}_mr"] = [np.sum([(y in str(x).lower())*1 for x in reduced_data[reduced_data["Model"]=="Mr"]["Subtype"]]) for y in subtypes]

            ax = sb.countplot(data=reduced_data, x="gleason_grade_pattern_1", hue="Model", palette=['b', 'r'])
            for container in ax.containers:
                ax.bar_label(container)
            plt.title(f"{quad}_{datatype}")
            ax.spines[['right', 'top']].set_visible(False)
            ax.legend(frameon=False)
            plt.xlabel("Gleason Grade Pattern")
            plt.ylabel("# of Tiles")
            plt.ylim([0, 200])
            plt.savefig(f"{quad}_{datatype}" + ".png", dpi=600)
            plt.show()


    subtype_df = pd.DataFrame(subtype_dict)
    # subtype_df.to_excel("PathSubtypes2.xlsx")

if by_grade:
    # Conduct chi squared test and create other bar plots
    master_data = master_data.dropna(subset=["gleason_grade_pattern_1"])
    chi_squared_dict = {"RP":[], "BX":[]}
    grade_list = []
    for grade in pd.unique(master_data["gleason_grade_pattern_1"]):
        grade_list.append(grade)
        for datatype in master_data["Data"].unique():
            reduced_data = master_data[master_data["gleason_grade_pattern_1"] == grade]
            reduced_data = reduced_data[reduced_data["Data"] == datatype]

            if grade in ["No Cancer"]:
                reduced_data = reduced_data[[x in ["FP", "TN"] for x in reduced_data["result"]]]
            if grade in [4.0, 3.0]:
                reduced_data = reduced_data[[x in ["TP", "FN"] for x in reduced_data["result"]]]

            contingency_table = pd.crosstab(reduced_data['Model'], reduced_data['result'])
            chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
            chi_squared_dict[datatype].append(p)

            ax = sb.countplot(data=reduced_data, x="result", hue="Model", palette=['b', 'r'])
            for container in ax.containers:
                ax.bar_label(container)
            plt.title(f"Grade: {grade} | {datatype} | Chi^2: {p}")
            ax.spines[['right', 'top']].set_visible(False)
            ax.legend(frameon=False)
            plt.xlabel("Result")
            plt.ylabel("# of Tiles")
            plt.ylim([0, 200])
            # plt.savefig(f"{grade}_{datatype}" + ".png", dpi=600)
            plt.show()

            bla = 1

    chi_squared_dict["Grade"] = grade_list
    chi_squared_df = pd.DataFrame(chi_squared_dict)
    chi_squared_df.to_excel("ChiSquaredTests.xlsx")

