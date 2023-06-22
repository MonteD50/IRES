import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

variables = ['handicapped_infants_n', 'handicapped_infants_y',
       'water_project_cost_sharing_n', 'water_project_cost_sharing_y',
       'adoption_of_the_budget_resolution_n',
       'adoption_of_the_budget_resolution_y', 'physician_fee_freeze_n',
       'physician_fee_freeze_y', 'el_salvador_aid_n', 'el_salvador_aid_y',
       'religious_groups_in_schools_n', 'religious_groups_in_schools_y',
       'anti_satellite_test_ban_n', 'anti_satellite_test_ban_y',
       'aid_to_nicaraguan_contras_n', 'aid_to_nicaraguan_contras_y',
       'mx_missile_n', 'mx_missile_y', 'immigration_n', 'immigration_y',
       'synfuels_corporation_cutback_n', 'synfuels_corporation_cutback_y',
       'education_spending_n', 'education_spending_y',
       'superfund_right_to_sue_n', 'superfund_right_to_sue_y', 'crime_n',
       'crime_y', 'duty_free_exports_n', 'duty_free_exports_y',
       'export_administration_act_south_africa_n',
       'export_administration_act_south_africa_y']

car = pd.read_csv("class_association_rules.csv")

# Unique values for 'right'
unique_right = car['right'].unique()

for class_name in unique_right:
    df_class = car[car['right'] == class_name]
    res = pd.DataFrame(columns = variables)
    # Loop through df_class
    for index, row in df_class.iterrows():
        left = row['left']
        res_row = {}
        for variable in variables:
            if variable in left:
                label = 1
            else:
                label = 0
            res_row[variable] = label
        row_series = pd.Series(res_row, name = left)
        res = res.append(row_series, ignore_index = False)
    res.to_csv("binary_distance_" + class_name + ".csv")