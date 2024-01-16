import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle


class ParseResults(object):

    def __init__(self):
        pass

    def parse_extra_mutated_scenarios(self):

        road_list = ["road1", "road2", "road3", "road4"]
        weather_list = ["rain_day", "rain_night", "sunny_day", "clear_night"]
        parameter_list = ["original", "position", "rotation", "velocity", "angular_velocity"]
        model_list = ["gpt-3.5-turbo-1106", "llama-v2-13b-chat", "mistral-7b-instruct-4k"]
        extra_list = ["increase", "decrease"]
        repeat_time = 10

        for road in road_list:
            for weather in weather_list:
                evaluate_R_MR_extra_4full_experiments_path = "./outputs_results/" + road + "-" + weather + "-scenarios/"
                if not os.path.exists(evaluate_R_MR_extra_4full_experiments_path):
                    continue
                scenarios_names_list = [d for d in os.listdir(evaluate_R_MR_extra_4full_experiments_path) if os.path.isdir(
                    os.path.join(evaluate_R_MR_extra_4full_experiments_path, d))]
                for scenario_name in scenarios_names_list:
                    for index_parameter, parameter in enumerate(parameter_list):
                        for extra in extra_list:
                            csv_data = {}
                            csv_model = []
                            csv_temperature_repeat = []
                            csv_0_0 = []
                            csv_0_5 = []
                            csv_1_0 = []
                            csv_1_5 = []
                            csv_2_0 = []
                            csv_2_5 = []
                            csv_3_0 = []
                            csv_realistic = []
                            csv_rea_pro = []
                            csv_rea_cond = []
                            csv_scenario = []
                            csv_sce_pro = []
                            csv_sce_cond = []
                            csv_time = []
                            for model in model_list:
                                if model == "gpt-3.5-turbo-1106":
                                    temperature = 0
                                else:
                                    temperature = 0
                                for index in range(0, repeat_time):
                                    if parameter == "original":
                                        if extra == "increase":
                                            results_file_name = f"{evaluate_R_MR_extra_4full_experiments_path}{scenario_name}/{parameter}/{model}_{index}.txt"
                                        else:
                                            continue
                                    else:
                                        results_file_name = f"{evaluate_R_MR_extra_4full_experiments_path}{scenario_name}/{parameter}_{extra}/{model}_{index}.txt"
                                    with open(results_file_name, 'r') as file:
                                        results = file.read()
                                    index_before = results.rfind('{')
                                    index_after = results.rfind('}') + 1
                                    json_results = results[index_before:index_after]
                                    json_results = json_results.lower()
                                    json_results = json_results.replace("%", "").replace("/10.0", "").replace("/10",
                                                                                                              "").replace(
                                        "\\", "").replace("\'", "\"").replace(",\n}", "\n}")
                                    if '"realistic": 0.0' in json_results:
                                        json_results = json_results.replace('"realistic": 0.0', '"realistic": 999')
                                    elif '"realistic": 1.0' in json_results:
                                        json_results = json_results.replace('"realistic": 1.0', '"realistic": 999')
                                    json_results = json_results.splitlines()
                                    for i in range(1, len(json_results) - 2):
                                        if not json_results[i].endswith(","):
                                            json_results[i] += ","
                                    json_results = '\n'.join(json_results)

                                    index_before = results.rfind("create_time: ")
                                    index_after = results.rfind(" output_time")
                                    time_results = results[index_before:index_after].split("create_time: ")[-1].split("s")[
                                        0]
                                    csv_time.append(float(time_results))

                                    try:
                                        json_results = json.loads(json_results)
                                    except json.JSONDecodeError as e:
                                        csv_model.append(model)
                                        csv_temperature_repeat.append(f"{str(temperature)}_{index}")
                                        csv_0_0.append("")
                                        csv_0_5.append("")
                                        csv_1_0.append("")
                                        csv_1_5.append("")
                                        csv_2_0.append("")
                                        csv_2_5.append("")
                                        csv_3_0.append("")
                                        if "Realistic: True\n" in results or "Realistic:\nTrue\n" in results:
                                            csv_realistic.append(True)
                                        elif "Realistic: False\n" in results or "Realistic:\nFalse\n" in results:
                                            csv_realistic.append(False)
                                        else:
                                            csv_realistic.append("")
                                        csv_rea_pro.append("")
                                        csv_rea_cond.append("")
                                        csv_scenario.append("")
                                        csv_sce_pro.append("")
                                        csv_sce_cond.append("")
                                        continue
                                    csv_model.append(model)
                                    csv_temperature_repeat.append(f"{str(temperature)}_{index}")
                                    csv_0_0.append(json_results["0.0 seconds"] if "0.0 seconds" in json_results else "")
                                    csv_0_5.append(json_results["0.5 seconds"] if "0.5 seconds" in json_results else "")
                                    csv_1_0.append(json_results["1.0 seconds"] if "1.0 seconds" in json_results else "")
                                    csv_1_5.append(json_results["1.5 seconds"] if "1.5 seconds" in json_results else "")
                                    csv_2_0.append(json_results["2.0 seconds"] if "2.0 seconds" in json_results else "")
                                    csv_2_5.append(json_results["2.5 seconds"] if "2.5 seconds" in json_results else "")
                                    csv_3_0.append(json_results["3.0 seconds"] if "3.0 seconds" in json_results else "")
                                    if "realistic" in json_results and (
                                            json_results["realistic"] == True or json_results["realistic"] == "true" or
                                            json_results["realistic"] == "True" or json_results["realistic"] == False or
                                            json_results["realistic"] == "false" or json_results[
                                                "realistic"] == "False"):
                                        csv_realistic.append(json_results["realistic"])
                                    elif "Realistic: True\n" in results or "Realistic:\nTrue\n" in results:
                                        csv_realistic.append(True)
                                    elif "Realistic: False\n" in results or "Realistic:\nFalse\n" in results:
                                        csv_realistic.append(False)
                                    else:
                                        csv_realistic.append("")
                                    csv_rea_pro.append(json_results[
                                                           "realistic_probability"] if "realistic_probability" in json_results else "")
                                    csv_rea_cond.append(json_results[
                                                            "realistic_confidence"] if "realistic_confidence" in json_results else "")
                                    csv_scenario.append(json_results["scenario"] if "scenario" in json_results else "")
                                    csv_sce_pro.append(json_results[
                                                           "scenario_probability"] if "scenario_probability" in json_results else "")
                                    csv_sce_cond.append(json_results[
                                                            "scenario_confidence"] if "scenario_confidence" in json_results else "")
                            csv_data["Model"] = csv_model
                            csv_data["Temperature_repeatIndex"] = csv_temperature_repeat
                            csv_data["0.0s"] = csv_0_0
                            csv_data["0.5s"] = csv_0_5
                            csv_data["1.0s"] = csv_1_0
                            csv_data["1.5s"] = csv_1_5
                            csv_data["2.0s"] = csv_2_0
                            csv_data["2.5s"] = csv_2_5
                            csv_data["3.0s"] = csv_3_0
                            csv_data["Realistic"] = csv_realistic
                            csv_data["Realistic_Probability"] = csv_rea_pro
                            csv_data["Realistic_Confidence"] = csv_rea_cond
                            csv_data["Scenario"] = csv_scenario
                            csv_data["Scenario_Probability"] = csv_sce_pro
                            csv_data["Scenario_Confidence"] = csv_sce_cond
                            csv_data["Time"] = csv_time
                            df = pd.DataFrame(csv_data)
                            if parameter == "original":
                                if extra == "increase":
                                    csv_file_path = evaluate_R_MR_extra_4full_experiments_path + f"{scenario_name}_{parameter}.csv"
                                else:
                                    continue
                            else:
                                csv_file_path = evaluate_R_MR_extra_4full_experiments_path + f"{scenario_name}_{parameter}_{extra}.csv"
                            df.to_csv(csv_file_path, index=False)

    def count_extra_mutated_scenarios_robustness_RQ(self):

        road_list = ["road1", "road2", "road3", "road4"]
        road_name_list = ["Road1", "Road2", "Road3", "Road4"]
        weather_list = ["rain_day", "rain_night", "sunny_day", "clear_night"]
        weather_name_list = ["Rainy Day", "Rainy Night", "Sunny Day", "Clear Night"]
        parameter_list = ["original", "position", "rotation", "velocity", "angular_velocity"]
        model_list = ["gpt-3.5-turbo-1106", "llama-v2-13b-chat", "mistral-7b-instruct-4k"]
        extra_list = ["increase", "decrease"]
        scenario_number = 64

        RQ_df = pd.DataFrame()
        RQ_rank_df = pd.DataFrame()

        for model in model_list:
            if model == "gpt-3.5-turbo-1106":
                temperature = 0
            else:
                temperature = 0

            scenario_index = 0

            for index_road, road in enumerate(road_list):
                for index_weather, weather in enumerate(weather_list):
                    evaluate_R_MR_extra_4full_experiments_path = "./outputs_results/" + road + "-" + weather + "-scenarios/"
                    if not os.path.exists(evaluate_R_MR_extra_4full_experiments_path):
                        continue
                    scenarios_names_list = [d for d in os.listdir(evaluate_R_MR_extra_4full_experiments_path) if
                                            os.path.isdir(
                                                os.path.join(evaluate_R_MR_extra_4full_experiments_path, d))]
                    for scenario_name in scenarios_names_list:
                        scenario_index += 1

                        scenario_matched_count = [0 for _ in range(len(parameter_list) - 1)]
                        scenario_success_average = [0 for _ in range(len(parameter_list) - 1)]
                        scenario_diff_count = [0 for _ in range(len(parameter_list) - 1)]
                        scenario_robustness_score = [0 for _ in range(len(parameter_list) - 1)]

                        matched_true_count = {}
                        matched_false_count = {}
                        for index, parameter in enumerate(parameter_list):
                            for extra in extra_list:
                                if parameter == "original":
                                    if extra == "increase":
                                        results = pd.read_csv(
                                            evaluate_R_MR_extra_4full_experiments_path + scenario_name + "_" + parameter + ".csv")
                                    else:
                                        continue
                                else:
                                    results = pd.read_csv(
                                        evaluate_R_MR_extra_4full_experiments_path + scenario_name + "_" + parameter + "_" + extra + ".csv")

                                rows = results[
                                    (results['Model'] == model) & results['Temperature_repeatIndex'].str.contains(
                                        f'{str(temperature)}_')]
                                true_count = rows['Realistic'].isin([True, 'True', 'true', 'TRUE']).sum()
                                false_count = rows['Realistic'].isin([False, 'False', 'false', 'FALSE']).sum()
                                inconclusive_count = 10 - true_count - false_count

                                matched_true_count[
                                    parameter if parameter == "original" else f"{parameter}_{extra}"] = true_count
                                matched_false_count[
                                    parameter if parameter == "original" else f"{parameter}_{extra}"] = false_count

                        for index, parameter in enumerate(parameter_list):
                            if parameter == "original":
                                continue
                            for extra in extra_list:
                                scenario_matched_count[index - 1] += min(matched_true_count["original"], matched_true_count[f"{parameter}_{extra}"])
                                scenario_success_average[index - 1] += (matched_true_count["original"] + matched_true_count[f"{parameter}_{extra}"]) / 2.0
                                scenario_diff_count[index - 1] += abs(matched_true_count["original"] - matched_true_count[f"{parameter}_{extra}"])
                            scenario_matched_count[index - 1] /= len(extra_list)
                            scenario_success_average[index - 1] /= len(extra_list)
                            scenario_diff_count[index - 1] /= len(extra_list)
                            scenario_robustness_score[index - 1] = scenario_matched_count[index - 1] + \
                                                                   scenario_success_average[index - 1] - \
                                                                   scenario_diff_count[index - 1]

                        RQ_rank_row = {"Scenario": scenario_name, "Model": model, "Road": road_name_list[index_road],
                                      "Weather": weather_name_list[index_weather], "Scenario_ID": f'{scenario_index}'}
                        overall_robustness = 0
                        for index, parameter in enumerate(["Position", "Rotation", "Velocity", "Angular Velocity"]):
                            RQ_row = {"Scenario": scenario_name, "Model": model, "Road": road_name_list[index_road],
                                      "Weather": weather_name_list[index_weather], "Scenario_ID": f'{scenario_index}',
                                      "Parameter": parameter}
                            RQ_row["Matched"] = scenario_matched_count[index]
                            RQ_row["Success"] = scenario_success_average[index]
                            RQ_row["Difference"] = scenario_diff_count[index]
                            RQ_row["Robustness"] = scenario_robustness_score[index]
                            RQ_df = pd.concat([RQ_df, pd.DataFrame([RQ_row])], ignore_index=True)

                            RQ_rank_row[parameter] = scenario_robustness_score[index]
                            overall_robustness += scenario_robustness_score[index]
                        RQ_rank_row["Overall"] = overall_robustness / 4.0
                        RQ_rank_df = pd.concat([RQ_rank_df, pd.DataFrame([RQ_rank_row])], ignore_index=True)

        RQ_df.to_csv(f"./outputs_results/RQ_robustness.csv", index=False)
        RQ_rank_df.to_csv(f"./outputs_results/RQ_rank_robustness.csv", index=False)

    def count_extra_mutated_scenarios_robustness_RQ1(self):

        road_name_list = ["Road1", "Road2", "Road3", "Road4"]
        weather_name_list = ["Rainy Day", "Rainy Night", "Sunny Day", "Clear Night"]
        parameter_list = ["Position", "Rotation", "Velocity", "Angular Velocity"]
        model_list = ["gpt-3.5-turbo-1106", "llama-v2-13b-chat", "mistral-7b-instruct-4k"]
        scenario_number = 64

        data = pd.read_csv("./outputs_results/RQ_robustness.csv")

        RQ_a_df = pd.DataFrame()
        RQ_b_df = pd.DataFrame()
        RQ_c_df = pd.DataFrame()
        RQ_d_df = pd.DataFrame()

        for model in model_list:
            model_data = data[data["Model"] == model]
            RQ_a_matched = model_data["Matched"].mean()
            RQ_a_success = model_data["Success"].mean()
            RQ_a_diff = model_data["Difference"].mean()
            RQ_a_robustness = model_data["Robustness"].mean()
            RQ_a_row = {"Model": model, "Matched": RQ_a_matched, "Success": RQ_a_success,
                        "Difference": RQ_a_diff, "Robustness": RQ_a_robustness}
            RQ_a_df = pd.concat([RQ_a_df, pd.DataFrame([RQ_a_row])], ignore_index=True)
            for parameter in parameter_list:
                parameter_data = model_data[model_data["Parameter"] == parameter]
                RQ_c_matched = parameter_data["Matched"].mean()
                RQ_c_success = parameter_data["Success"].mean()
                RQ_c_diff = parameter_data["Difference"].mean()
                RQ_c_robustness = parameter_data["Robustness"].mean()
                RQ_c_row = {"Model": model, "Parameter": parameter, "Matched": RQ_c_matched,
                            "Success": RQ_c_success, "Difference": RQ_c_diff, "Robustness": RQ_c_robustness}
                RQ_c_df = pd.concat([RQ_c_df, pd.DataFrame([RQ_c_row])], ignore_index=True)
            for scenario_id in range(1, scenario_number + 1):
                scenario_data = model_data[model_data["Scenario_ID"] == scenario_id]
                RQ_b_matched = scenario_data["Matched"].mean()
                RQ_b_success = scenario_data["Success"].mean()
                RQ_b_diff = scenario_data["Difference"].mean()
                RQ_b_robustness = scenario_data["Robustness"].mean()
                RQ_b_row = {"Model": model, "Scenario_ID": scenario_id, "Matched": RQ_b_matched,
                            "Success": RQ_b_success, "Difference": RQ_b_diff, "Robustness": RQ_b_robustness}
                RQ_b_df = pd.concat([RQ_b_df, pd.DataFrame([RQ_b_row])], ignore_index=True)
                for parameter in parameter_list:
                    parameter_data = scenario_data[scenario_data["Parameter"] == parameter]
                    RQ_d_matched = parameter_data["Matched"].mean()
                    RQ_d_success = parameter_data["Success"].mean()
                    RQ_d_diff = parameter_data["Difference"].mean()
                    RQ_d_robustness = parameter_data["Robustness"].mean()
                    RQ_d_row = {"Model": model, "Scenario_ID": scenario_id, "Parameter": parameter, "Matched": RQ_d_matched,
                                "Success": RQ_d_success, "Difference": RQ_d_diff, "Robustness": RQ_d_robustness}
                    RQ_d_df = pd.concat([RQ_d_df, pd.DataFrame([RQ_d_row])], ignore_index=True)

        RQ_a_df.to_csv(f"./outputs_results/RQ1_a_robustness.csv", index=False)
        RQ_b_df.to_csv(f"./outputs_results/RQ1_b_robustness.csv", index=False)
        RQ_c_df.to_csv(f"./outputs_results/RQ1_c_robustness.csv", index=False)
        RQ_d_df.to_csv(f"./outputs_results/RQ1_d_robustness.csv", index=False)

    def count_extra_mutated_scenarios_robustness_RQ2(self):

        road_name_list = ["Road1", "Road2", "Road3", "Road4"]
        weather_name_list = ["Rainy Day", "Rainy Night", "Sunny Day", "Clear Night"]
        parameter_list = ["Position", "Rotation", "Velocity", "Angular Velocity"]
        model_list = ["gpt-3.5-turbo-1106", "llama-v2-13b-chat", "mistral-7b-instruct-4k"]
        scenario_number = 64

        data = pd.read_csv("./outputs_results/RQ_robustness.csv")

        RQ_a_df = pd.DataFrame()
        RQ_b_df = pd.DataFrame()
        RQ_c_df = pd.DataFrame()
        RQ_d_df = pd.DataFrame()

        for model in model_list:
            scenario_index = 1
            for road in road_name_list:
                model_data = data[(data["Model"] == model) & (data["Road"] == road)]
                RQ_a_matched = model_data["Matched"].mean()
                RQ_a_success = model_data["Success"].mean()
                RQ_a_diff = model_data["Difference"].mean()
                RQ_a_robustness = model_data["Robustness"].mean()
                RQ_a_row = {"Model": model, "Road": road, "Matched": RQ_a_matched, "Success": RQ_a_success,
                            "Difference": RQ_a_diff, "Robustness": RQ_a_robustness}
                RQ_a_df = pd.concat([RQ_a_df, pd.DataFrame([RQ_a_row])], ignore_index=True)
                for parameter in parameter_list:
                    parameter_data = model_data[model_data["Parameter"] == parameter]
                    RQ_c_matched = parameter_data["Matched"].mean()
                    RQ_c_success = parameter_data["Success"].mean()
                    RQ_c_diff = parameter_data["Difference"].mean()
                    RQ_c_robustness = parameter_data["Robustness"].mean()
                    RQ_c_row = {"Model": model, "Road": road, "Parameter": parameter, "Matched": RQ_c_matched,
                                "Success": RQ_c_success, "Difference": RQ_c_diff, "Robustness": RQ_c_robustness}
                    RQ_c_df = pd.concat([RQ_c_df, pd.DataFrame([RQ_c_row])], ignore_index=True)
                scenario_range = scenario_index + 16
                for scenario_id in range(scenario_index, scenario_range):
                    scenario_data = model_data[model_data["Scenario_ID"] == scenario_id]
                    RQ_b_matched = scenario_data["Matched"].mean()
                    RQ_b_success = scenario_data["Success"].mean()
                    RQ_b_diff = scenario_data["Difference"].mean()
                    RQ_b_robustness = scenario_data["Robustness"].mean()
                    RQ_b_row = {"Model": model, "Road": road, "Scenario_ID": scenario_id, "Matched": RQ_b_matched,
                                "Success": RQ_b_success, "Difference": RQ_b_diff, "Robustness": RQ_b_robustness}
                    RQ_b_df = pd.concat([RQ_b_df, pd.DataFrame([RQ_b_row])], ignore_index=True)
                    for parameter in parameter_list:
                        parameter_data = scenario_data[scenario_data["Parameter"] == parameter]
                        RQ_d_matched = parameter_data["Matched"].mean()
                        RQ_d_success = parameter_data["Success"].mean()
                        RQ_d_diff = parameter_data["Difference"].mean()
                        RQ_d_robustness = parameter_data["Robustness"].mean()
                        RQ_d_row = {"Model": model, "Road": road, "Scenario_ID": scenario_id, "Parameter": parameter,
                                    "Matched": RQ_d_matched, "Success": RQ_d_success, "Difference": RQ_d_diff,
                                    "Robustness": RQ_d_robustness}
                        RQ_d_df = pd.concat([RQ_d_df, pd.DataFrame([RQ_d_row])], ignore_index=True)
                scenario_index += 16

        RQ_a_df.to_csv(f"./outputs_results/RQ2_a_robustness.csv", index=False)
        RQ_b_df.to_csv(f"./outputs_results/RQ2_b_robustness.csv", index=False)
        RQ_c_df.to_csv(f"./outputs_results/RQ2_c_robustness.csv", index=False)
        RQ_d_df.to_csv(f"./outputs_results/RQ2_d_robustness.csv", index=False)

    def count_extra_mutated_scenarios_robustness_RQ3(self):

        road_name_list = ["Road1", "Road2", "Road3", "Road4"]
        weather_name_list = ["Rainy Day", "Rainy Night", "Sunny Day", "Clear Night"]
        parameter_list = ["Position", "Rotation", "Velocity", "Angular Velocity"]
        model_list = ["gpt-3.5-turbo-1106", "llama-v2-13b-chat", "mistral-7b-instruct-4k"]
        scenario_number = 64

        data = pd.read_csv("./outputs_results/RQ_robustness.csv")

        RQ_a_df = pd.DataFrame()
        RQ_b_df = pd.DataFrame()
        RQ_c_df = pd.DataFrame()
        RQ_d_df = pd.DataFrame()

        for model in model_list:
            scenario_index = 1
            for weather in weather_name_list:
                model_data = data[(data["Model"] == model) & (data["Weather"] == weather)]
                RQ_a_matched = model_data["Matched"].mean()
                RQ_a_success = model_data["Success"].mean()
                RQ_a_diff = model_data["Difference"].mean()
                RQ_a_robustness = model_data["Robustness"].mean()
                RQ_a_row = {"Model": model, "Weather": weather, "Matched": RQ_a_matched, "Success": RQ_a_success,
                            "Difference": RQ_a_diff, "Robustness": RQ_a_robustness}
                RQ_a_df = pd.concat([RQ_a_df, pd.DataFrame([RQ_a_row])], ignore_index=True)
                for parameter in parameter_list:
                    parameter_data = model_data[model_data["Parameter"] == parameter]
                    RQ_c_matched = parameter_data["Matched"].mean()
                    RQ_c_success = parameter_data["Success"].mean()
                    RQ_c_diff = parameter_data["Difference"].mean()
                    RQ_c_robustness = parameter_data["Robustness"].mean()
                    RQ_c_row = {"Model": model, "Weather": weather, "Parameter": parameter, "Matched": RQ_c_matched,
                                "Success": RQ_c_success, "Difference": RQ_c_diff, "Robustness": RQ_c_robustness}
                    RQ_c_df = pd.concat([RQ_c_df, pd.DataFrame([RQ_c_row])], ignore_index=True)

                start_index = scenario_index
                for _ in range(4):
                    for scenario_id in range(start_index, start_index + 4):
                        scenario_data = model_data[model_data["Scenario_ID"] == scenario_id]
                        RQ_b_matched = scenario_data["Matched"].mean()
                        RQ_b_success = scenario_data["Success"].mean()
                        RQ_b_diff = scenario_data["Difference"].mean()
                        RQ_b_robustness = scenario_data["Robustness"].mean()
                        RQ_b_row = {"Model": model, "Weather": weather, "Scenario_ID": scenario_id, "Matched": RQ_b_matched,
                                    "Success": RQ_b_success, "Difference": RQ_b_diff, "Robustness": RQ_b_robustness}
                        RQ_b_df = pd.concat([RQ_b_df, pd.DataFrame([RQ_b_row])], ignore_index=True)
                        for parameter in parameter_list:
                            parameter_data = scenario_data[scenario_data["Parameter"] == parameter]
                            RQ_d_matched = parameter_data["Matched"].mean()
                            RQ_d_success = parameter_data["Success"].mean()
                            RQ_d_diff = parameter_data["Difference"].mean()
                            RQ_d_robustness = parameter_data["Robustness"].mean()
                            RQ_d_row = {"Model": model, "Weather": weather, "Scenario_ID": scenario_id, "Parameter": parameter,
                                        "Matched": RQ_d_matched, "Success": RQ_d_success, "Difference": RQ_d_diff, "Robustness": RQ_d_robustness}
                            RQ_d_df = pd.concat([RQ_d_df, pd.DataFrame([RQ_d_row])], ignore_index=True)
                    start_index += 16
                scenario_index += 4

        RQ_a_df.to_csv(f"./outputs_results/RQ3_a_robustness.csv", index=False)
        RQ_b_df.to_csv(f"./outputs_results/RQ3_b_robustness.csv", index=False)
        RQ_c_df.to_csv(f"./outputs_results/RQ3_c_robustness.csv", index=False)
        RQ_d_df.to_csv(f"./outputs_results/RQ3_d_robustness.csv", index=False)

    def sort_robustness_RQ_a_c(self):

        model_list = ["gpt-3.5-turbo-1106", "llama-v2-13b-chat", "mistral-7b-instruct-4k"]
        model_name_list = ["GPT-3.5", "Llama2-13B", "Mistral-7B"]
        road_name_list = ["Road1", "Road2", "Road3", "Road4"]
        weather_name_list = ["Rainy Day", "Rainy Night", "Sunny Day", "Clear Night"]
        parameter_list = ["Position", "Rotation", "Velocity", "Angular Velocity"]
        parameter_name_list = ["Overall", "Position", "Rotation", "Velocity", "Angular Velocity"]
        scenario_number = 64

        output_df = pd.DataFrame()

        overall_df = pd.read_csv("./outputs_results/RQ1_a_robustness.csv")
        parameter_df = pd.read_csv("./outputs_results/RQ1_c_robustness.csv")
        row = {"RQ": "RQ1", "Category": "All"}
        for index, model in enumerate(model_list):
            overall_model_data = overall_df[overall_df["Model"] == model]
            parameter_model_data = parameter_df[parameter_df["Model"] == model]
            row[f"{model_name_list[index][0]}_All"] = str(round(overall_model_data["Robustness"].tolist()[0], 2))
            for parameter in parameter_list:
                parameter_data = parameter_model_data[parameter_model_data["Parameter"] == parameter]["Robustness"].tolist()[0]
                row[f"{model_name_list[index][0]}_{parameter[0]}"] = str(round(parameter_data, 2))
        output_df = pd.concat([output_df, pd.DataFrame([row])], ignore_index=True)

        overall_df = pd.read_csv("./outputs_results/RQ2_a_robustness.csv")
        parameter_df = pd.read_csv("./outputs_results/RQ2_c_robustness.csv")
        row = {"RQ": "RQ2"}
        for road in road_name_list:
            row["Category"] = road
            for index, model in enumerate(model_list):
                overall_model_data = overall_df[(overall_df["Model"] == model) & (overall_df["Road"] == road)]
                parameter_model_data = parameter_df[(parameter_df["Model"] == model) & (parameter_df["Road"] == road)]
                row[f"{model_name_list[index][0]}_All"] = str(round(overall_model_data["Robustness"].tolist()[0], 2))
                for parameter in parameter_list:
                    parameter_data = \
                    parameter_model_data[parameter_model_data["Parameter"] == parameter]["Robustness"].tolist()[0]
                    row[f"{model_name_list[index][0]}_{parameter[0]}"] = str(round(parameter_data, 2))
            output_df = pd.concat([output_df, pd.DataFrame([row])], ignore_index=True)

        overall_df = pd.read_csv("./outputs_results/RQ3_a_robustness.csv")
        parameter_df = pd.read_csv("./outputs_results/RQ3_c_robustness.csv")
        row = {"RQ": "RQ3"}
        for weather in weather_name_list:
            row["Category"] = weather
            for index, model in enumerate(model_list):
                overall_model_data = overall_df[(overall_df["Model"] == model) & (overall_df["Weather"] == weather)]
                parameter_model_data = parameter_df[(parameter_df["Model"] == model) & (parameter_df["Weather"] == weather)]
                row[f"{model_name_list[index][0]}_All"] = str(round(overall_model_data["Robustness"].tolist()[0], 2))
                for parameter in parameter_list:
                    parameter_data = \
                        parameter_model_data[parameter_model_data["Parameter"] == parameter]["Robustness"].tolist()[0]
                    row[f"{model_name_list[index][0]}_{parameter[0]}"] = str(round(parameter_data, 2))
            output_df = pd.concat([output_df, pd.DataFrame([row])], ignore_index=True)

        output_df.to_csv(f"./outputs_results/RQ_a_c_robustness.csv", index=False)

    def sort_rank_robustness_RQ_b_d(self):

        model_list = ["gpt-3.5-turbo-1106", "llama-v2-13b-chat", "mistral-7b-instruct-4k"]
        model_name_list = ["GPT-3.5", "Llama2-13B", "Mistral-7B"]
        road_name_list = ["Road1", "Road2", "Road3", "Road4"]
        weather_name_list = ["Rainy Day", "Rainy Night", "Sunny Day", "Clear Night"]
        parameter_list = ["Position", "Rotation", "Velocity", "Angular Velocity"]
        parameter_name_list = ["Overall", "Position", "Rotation", "Velocity", "Angular Velocity"]
        scenario_number = 64

        df = pd.read_csv("./outputs_results/RQ_rank_robustness.csv")

        output_df = pd.DataFrame()
        for parameter in parameter_name_list:
            top_output = []
            bottom_output = []
            for model in model_list:
                model_data = df[df['Model'] == model]
                for sign in ["Top", "Bottom"]:
                    if sign == "Top":
                        values = model_data[parameter].drop_duplicates().nlargest(5)
                    else:
                        values = model_data[parameter].drop_duplicates().nsmallest(5)
                    output = []
                    for index, value in enumerate(values):
                        selected_rows = model_data[model_data[parameter] == value]
                        output += selected_rows['Scenario_ID'].tolist()
                    if sign == "Top":
                        top_output.append(output)
                    else:
                        bottom_output.append(output)
            row = {"Parameter": parameter}
            model_set1 = set(top_output[0])
            model_set2 = set(top_output[1])
            model_set3 = set(top_output[2])
            row["Top_All"] = str(len(model_set1.intersection(model_set2, model_set3)))
            for i in range(0, len(model_list) - 1):
                model_top_set1 = set(top_output[i])
                for j in range(i + 1, len(model_list)):
                    model_top_set2 = set(top_output[j])
                    row[f"Top_{model_name_list[i][0]}-{model_name_list[j][0]}"] = str(len(model_top_set1.intersection(model_top_set2)))
            model_set1 = set(bottom_output[0])
            model_set2 = set(bottom_output[1])
            model_set3 = set(bottom_output[2])
            row["Bottom_All"] = str(len(model_set1.intersection(model_set2, model_set3)))
            for i in range(0, len(model_list) - 1):
                model_bottom_set1 = set(bottom_output[i])
                for j in range(i + 1, len(model_list)):
                    model_bottom_set2 = set(bottom_output[j])
                    row[f"Bottom_{model_name_list[i][0]}-{model_name_list[j][0]}"] = str(len(model_bottom_set1.intersection(model_bottom_set2)))
            for i in range(0, len(model_list) - 1):
                model_top_set1 = set(top_output[i])
                model_bottom_set1 = set(bottom_output[i])
                for j in range(i + 1, len(model_list)):
                    model_top_set2 = set(top_output[j])
                    model_bottom_set2 = set(bottom_output[j])
                    row[f"Inc_{model_name_list[i][0]}-{model_name_list[j][0]}"] = str(len(model_top_set1.intersection(model_bottom_set2)) + len(model_top_set2.intersection(model_bottom_set1)))
            output_df = pd.concat([output_df, pd.DataFrame([row])], ignore_index=True)
        output_df.to_csv(f"./outputs_results/RQ1_b_d_rank_robustness.csv", index=False)

        output_df = pd.DataFrame()
        for road in road_name_list:
            for parameter in parameter_name_list:
                top_output = []
                bottom_output = []
                for model in model_list:
                    model_data = df[(df['Model'] == model) & (df['Road'] == road)]
                    for sign in ["Top", "Bottom"]:
                        if sign == "Top":
                            values = model_data[parameter].drop_duplicates().nlargest(5)
                        else:
                            values = model_data[parameter].drop_duplicates().nsmallest(5)
                        output = []
                        for index, value in enumerate(values):
                            selected_rows = model_data[model_data[parameter] == value]
                            output += selected_rows['Scenario_ID'].tolist()
                        if sign == "Top":
                            top_output.append(output)
                        else:
                            bottom_output.append(output)
                row = {"Road": road, "Parameter": parameter}
                model_set1 = set(top_output[0])
                model_set2 = set(top_output[1])
                model_set3 = set(top_output[2])
                row["Top_All"] = str(len(model_set1.intersection(model_set2, model_set3)))
                for i in range(0, len(model_list) - 1):
                    model_top_set1 = set(top_output[i])
                    for j in range(i + 1, len(model_list)):
                        model_top_set2 = set(top_output[j])
                        row[f"Top_{model_name_list[i][0]}-{model_name_list[j][0]}"] = str(
                            len(model_top_set1.intersection(model_top_set2)))
                model_set1 = set(bottom_output[0])
                model_set2 = set(bottom_output[1])
                model_set3 = set(bottom_output[2])
                row["Bottom_All"] = str(len(model_set1.intersection(model_set2, model_set3)))
                for i in range(0, len(model_list) - 1):
                    model_bottom_set1 = set(bottom_output[i])
                    for j in range(i + 1, len(model_list)):
                        model_bottom_set2 = set(bottom_output[j])
                        row[f"Bottom_{model_name_list[i][0]}-{model_name_list[j][0]}"] = str(
                            len(model_bottom_set1.intersection(model_bottom_set2)))
                for i in range(0, len(model_list) - 1):
                    model_top_set1 = set(top_output[i])
                    model_bottom_set1 = set(bottom_output[i])
                    for j in range(i + 1, len(model_list)):
                        model_top_set2 = set(top_output[j])
                        model_bottom_set2 = set(bottom_output[j])
                        row[f"Inc_{model_name_list[i][0]}-{model_name_list[j][0]}"] = str(
                            len(model_top_set1.intersection(model_bottom_set2)) + len(
                                model_top_set2.intersection(model_bottom_set1)))
                output_df = pd.concat([output_df, pd.DataFrame([row])], ignore_index=True)
        output_df.to_csv(f"./outputs_results/RQ2_b_d_rank_robustness.csv", index=False)

        output_df = pd.DataFrame()
        for weather in weather_name_list:
            for parameter in parameter_name_list:
                top_output = []
                bottom_output = []
                for model in model_list:
                    model_data = df[(df['Model'] == model) & (df['Weather'] == weather)]
                    for sign in ["Top", "Bottom"]:
                        if sign == "Top":
                            values = model_data[parameter].drop_duplicates().nlargest(5)
                        else:
                            values = model_data[parameter].drop_duplicates().nsmallest(5)
                        output = []
                        for index, value in enumerate(values):
                            selected_rows = model_data[model_data[parameter] == value]
                            output += selected_rows['Scenario_ID'].tolist()
                        if sign == "Top":
                            top_output.append(output)
                        else:
                            bottom_output.append(output)
                row = {"Weather": weather, "Parameter": parameter}
                model_set1 = set(top_output[0])
                model_set2 = set(top_output[1])
                model_set3 = set(top_output[2])
                row["Top_All"] = str(len(model_set1.intersection(model_set2, model_set3)))
                for i in range(0, len(model_list) - 1):
                    model_top_set1 = set(top_output[i])
                    for j in range(i + 1, len(model_list)):
                        model_top_set2 = set(top_output[j])
                        row[f"Top_{model_name_list[i][0]}-{model_name_list[j][0]}"] = str(
                            len(model_top_set1.intersection(model_top_set2)))
                model_set1 = set(bottom_output[0])
                model_set2 = set(bottom_output[1])
                model_set3 = set(bottom_output[2])
                row["Bottom_All"] = str(len(model_set1.intersection(model_set2, model_set3)))
                for i in range(0, len(model_list) - 1):
                    model_bottom_set1 = set(bottom_output[i])
                    for j in range(i + 1, len(model_list)):
                        model_bottom_set2 = set(bottom_output[j])
                        row[f"Bottom_{model_name_list[i][0]}-{model_name_list[j][0]}"] = str(
                            len(model_bottom_set1.intersection(model_bottom_set2)))
                for i in range(0, len(model_list) - 1):
                    model_top_set1 = set(top_output[i])
                    model_bottom_set1 = set(bottom_output[i])
                    for j in range(i + 1, len(model_list)):
                        model_top_set2 = set(top_output[j])
                        model_bottom_set2 = set(bottom_output[j])
                        row[f"Inc_{model_name_list[i][0]}-{model_name_list[j][0]}"] = str(
                            len(model_top_set1.intersection(model_bottom_set2)) + len(
                                model_top_set2.intersection(model_bottom_set1)))
                output_df = pd.concat([output_df, pd.DataFrame([row])], ignore_index=True)
        output_df.to_csv(f"./outputs_results/RQ3_b_d_rank_robustness.csv", index=False)

    def draw_violin_diagram_RQ0_robustness(self):

        model_list = ["gpt-3.5-turbo-1106", "llama-v2-13b-chat", "mistral-7b-instruct-4k"]
        model_name_list = ["GPT-3.5", "Llama2-13B", "Mistral-7B"]
        road_name_list = ["Road1", "Road2", "Road3", "Road4"]
        weather_name_list = ["Rainy Day", "Rainy Night", "Sunny Day", "Clear Night"]
        parameter_list = ["Position", "Rotation", "Velocity", "Angular Velocity"]
        parameter_name_list = ["Overall", "Position", "Rotation", "Velocity", "Angular Velocity"]
        color_list = [(169, 107, 92), (169, 107, 92), (169, 107, 92)]
        color_list = [(r / 255.0, g / 255.0, b / 255.0) for r, g, b in color_list]
        scenario_number = 64

        csv_file_path = "./outputs_results/RQ_robustness.csv"
        data = pd.read_csv(csv_file_path)

        model_data_chart = {"Model": [], "Value": []}

        for index, model in enumerate(model_list):
            model_data = data[data["Model"] == model]
            for _ in range(len(model_data)):
                model_data_chart["Model"].append(model_name_list[index])
            model_data_chart["Value"] += model_data["Success"].tolist()

        df = pd.DataFrame(model_data_chart)

        sns.set(style="whitegrid", rc={'font.size': 20, 'font.weight': 'bold'})
        plt.figure(figsize=(14, 6))
        ax = sns.violinplot(x='Model', y='Value', data=df, palette=color_list)
        ax_box = sns.boxplot(x="Model", y="Value", data=df, ax=ax, width=0.1, color="black",
                             showmeans=True, meanline=True, showfliers=False,
                             boxprops=dict(edgecolor=(38/255, 38/255, 38/255), facecolor=(38/255, 38/255, 38/255)),
                             capprops={'visible': False},
                             whiskerprops=dict(color=(38/255, 38/255, 38/255), linewidth=2),
                             medianprops={'visible': False},
                             meanprops={"color":"white", "linewidth":1, "linestyle": '-'})
        means = df.groupby("Model")["Value"].mean()
        for model, mean_value in means.items():
            ax_box.text(model, mean_value, f'{mean_value * 10:.1f}%', ha='center', va='bottom', color='white',
                        fontsize=20)

        ax.set(xlabel=None, ylabel="Success Rate")
        custom_labels = ["", "", "0%", "25%", "50%", "75%", "100%", "", ""]
        ax.set_yticklabels(custom_labels)
        ax.set_ylabel("Success Rate", fontsize=20, fontweight='bold')
        ax.tick_params(axis='y', labelsize=20)
        ax.tick_params(axis='x', labelsize=20)
        for i in range(len(model_list)):
            ax.get_children()[3 + 4 * i].set_visible(False)
            ax.get_children()[2 + 4 * i].set_visible(False)
            ax.get_children()[1 + 4 * i].set_visible(False)

        plt.savefig(f'RQ0_success_rate_violin_diagram.pdf',
                    bbox_inches="tight")
        plt.savefig(f'RQ0_success_rate_violin_diagram.svg',
                    format='svg', bbox_inches="tight")
        plt.tight_layout()
        plt.show()

    def draw_violin_diagram_RQ1_robustness(self):

        model_list = ["gpt-3.5-turbo-1106", "llama-v2-13b-chat", "mistral-7b-instruct-4k"]
        model_name_list = ["GPT-3.5", "Llama2-13B", "Mistral-7B"]
        road_name_list = ["Road1", "Road2", "Road3", "Road4"]
        weather_name_list = ["Rainy Day", "Rainy Night", "Sunny Day", "Clear Night"]
        parameter_list = ["Position", "Rotation", "Velocity", "Angular Velocity"]
        parameter_name_list = ["Overall", "Position", "Rotation", "Velocity", "Angular Velocity"]
        color_list = [(109, 47, 32), (183, 83, 71), (223, 126, 102), (237, 199, 117), (148, 181, 148), (34, 75, 94)]
        color_list = [(r / 255.0, g / 255.0, b / 255.0) for r, g, b in color_list]
        scenario_number = 64

        csv_file_path = "./outputs_results/RQ_robustness.csv"
        data = pd.read_csv(csv_file_path)

        model_data_chart = {"Model": [], "Parameter": [], "Value": []}

        for index, model in enumerate(model_list):
            model_data = data[data["Model"] == model]
            data_chart = {"Value": []}
            overall_list = []
            for scenario_id in range(1, scenario_number + 1):
                scenario_data = model_data[model_data["Scenario_ID"] == scenario_id]
                overall_list.append(scenario_data["Robustness"].mean())
            data_chart["Value"] += overall_list
            for parameter in parameter_list:
                param_data = model_data[model_data["Parameter"] == parameter]
                data_chart["Value"] += param_data["Robustness"].tolist()

            model_data_chart["Model"] += [model_name_list[index]] * 64 * len(parameter_name_list)
            model_data_chart["Parameter"] += np.repeat([param for param in parameter_name_list], 64).tolist()
            model_data_chart["Value"] += data_chart["Value"]

        model_data_chart_df = pd.DataFrame(model_data_chart)

        sns.set(style="whitegrid", rc={'font.size': 20, 'font.weight': 'bold'})
        g = sns.FacetGrid(model_data_chart_df, col="Model", col_wrap=3, height=5, sharey=True)
        g.map(sns.violinplot, "Parameter", "Value",
              order=[param for param in parameter_name_list],
              palette=sns.color_palette(color_list))

        for ax, title in zip(g.axes, model_data_chart_df['Model'].unique()):
            ax.set_title(f'{title}', fontsize=30, fontweight="bold")

            median_data = model_data_chart_df[model_data_chart_df['Model'] == title].groupby('Parameter')[
                'Value'].mean().reset_index()
            sns.pointplot(x='Parameter', y='Value', data=median_data, color='white', markers='D', join=False, ax=ax,
                          markersize=2)
            for i in range(len(parameter_name_list)):
                ax.get_children()[3 + 4 * i].set_visible(False)

        g.set_axis_labels("", "")
        for ax in g.axes:
            ax.set_xticklabels([])
            ax.set_xlabel("")

            ax.spines['left'].set_color('black')
            ax.spines['left'].set_linewidth(2)
            ax.spines['bottom'].set_color('black')
            ax.spines['bottom'].set_linewidth(2)
            ax.spines['right'].set_color('black')
            ax.spines['right'].set_linewidth(2)
            ax.spines['top'].set_color('black')
            ax.spines['top'].set_linewidth(2)

        for ax in g.axes.flat:
            ax.set_ylim(-15.0, 30.0)
            ax.set_yticks(np.arange(-15.0, 31.0, 5.0))
            ax.tick_params(axis='y', labelsize=18)

        g.axes.flat[0].annotate("Robustness Score", xy=(-0.15, 0.5), xycoords='axes fraction', ha='right', va='center',
                                fontsize=25, fontweight='bold', rotation=90)

        legend_elements = [Rectangle((0, 0), 1, 1, color=color_list[i], label=label, linewidth=0) for i, label in enumerate(parameter_name_list)]
        plt.legend(handles=legend_elements, bbox_to_anchor=(-0.55, 0.0), loc='upper center', ncol=6, fontsize=18)

        plt.savefig(f'RQ1_a_c_robustness_violin_diagram.pdf', bbox_inches="tight")
        plt.savefig(f'RQ1_a_c_robustness_violin_diagram.svg', format='svg', bbox_inches="tight")
        plt.tight_layout()
        plt.show()

    def draw_violin_diagram_RQ2_robustness(self):

        model_list = ["gpt-3.5-turbo-1106", "llama-v2-13b-chat", "mistral-7b-instruct-4k"]
        model_name_list = ["GPT-3.5", "Llama2-13B", "Mistral-7B"]
        road_name_list = ["Road1", "Road2", "Road3", "Road4"]
        weather_name_list = ["Rainy Day", "Rainy Night", "Sunny Day", "Clear Night"]
        parameter_list = ["Position", "Rotation", "Velocity", "Angular Velocity"]
        parameter_name_list = ["Overall", "Position", "Rotation", "Velocity", "Angular Velocity"]
        color_list = [(109, 47, 32), (183, 83, 71), (223, 126, 102), (237, 199, 117), (148, 181, 148), (34, 75, 94)]
        color_list = [(r / 255.0, g / 255.0, b / 255.0) for r, g, b in color_list]
        scenario_number = 64

        csv_file_path = "./outputs_results/RQ_robustness.csv"
        data = pd.read_csv(csv_file_path)

        model_data_chart = {"Model": [], "Road": [], "Parameter": [], "Value": []}

        for index, model in enumerate(model_list):
            scenario_index = 1
            for index_road, road in enumerate(road_name_list):
                model_data = data[(data['Model'] == model) & (data['Road'] == road)]
                data_chart = {"Value": []}

                overall_list = []
                scenario_range = scenario_index + 16
                for scenario_id in range(scenario_index, scenario_range):
                    scenario_data = model_data[model_data["Scenario_ID"] == scenario_id]
                    overall_list.append(scenario_data["Robustness"].mean())
                data_chart["Value"] += overall_list
                scenario_index += 16

                for index_param, parameter in enumerate(parameter_list):
                    param_data = model_data[model_data["Parameter"] == parameter]
                    data_chart["Value"] += param_data["Robustness"].tolist()

                model_data_chart["Model"] += [model_name_list[index]] * 16 * len(parameter_name_list)
                model_data_chart["Road"] += [road_name_list[index_road]] * 16 * len(parameter_name_list)
                model_data_chart["Parameter"] += np.repeat([param for param in parameter_name_list], (64 / len(road_name_list))).tolist()
                model_data_chart["Value"] += data_chart["Value"]

        model_data_chart_df = pd.DataFrame(model_data_chart)

        sns.set(style="whitegrid", rc={'font.size': 20, 'font.weight': 'bold'})
        g = sns.FacetGrid(model_data_chart_df, row="Road", col="Model", height=5, sharey=True)
        g.map(sns.violinplot, "Parameter", "Value",
              order=[param for param in parameter_name_list],
              palette=sns.color_palette(color_list))

        for i in range(len(road_name_list)):
            for ax, model_title in zip(g.axes[i], model_data_chart_df['Model'].unique()):
                median_data = model_data_chart_df[(model_data_chart_df['Model'] == model_title) & (
                        model_data_chart_df['Road'] == road_name_list[i])].groupby('Parameter')[
                    'Value'].mean().reset_index()
                sns.pointplot(x='Parameter', y='Value', data=median_data, color='white', markers='D', join=False, ax=ax,
                              markersize=2)
                for j in range(len(parameter_name_list)):
                    ax.get_children()[3 + 4 * j].set_visible(False)

        g.set_axis_labels("", "")

        for row_ax in g.axes:
            row_ax[-1].set_ylim(-20.0, 35.0)
            row_ax[-1].set_yticks(np.arange(-20.0, 36.0, 10.0))

        for ax in g.axes.flat:
            ax.set_xticklabels([])
            ax.set_xlabel("")
            ax.set_title("")

            for tick in ax.yaxis.get_major_ticks():
                tick.label1.set_fontsize(18)

            ax.spines['left'].set_color('black')
            ax.spines['left'].set_linewidth(2)
            ax.spines['bottom'].set_color('black')
            ax.spines['bottom'].set_linewidth(2)
            ax.spines['right'].set_color('black')
            ax.spines['right'].set_linewidth(2)
            ax.spines['top'].set_color('black')
            ax.spines['top'].set_linewidth(2)

        for ax, model_name in zip(g.axes[0], model_name_list):
            ax.annotate(model_name, xy=(0.5, 1.05), xycoords='axes fraction', ha='center', va='center',
                        fontsize=30, fontweight='bold')
        for ax, road_name in zip(g.axes[:, -1], road_name_list):
            ax.annotate(road_name, xy=(1.05, 0.5), xycoords='axes fraction', ha='left', va='center',
                        fontsize=30, fontweight='bold', rotation=-90)

        g.axes.flat[0].annotate("Robustness Score", xy=(-0.15, -1.2), xycoords='axes fraction', ha='right', va='center',
                                fontsize=30, fontweight='bold', rotation=90)

        legend_elements = [Rectangle((0, 0), 1, 1, color=color_list[i], label=label, linewidth=0) for i, label in enumerate(parameter_name_list)]
        plt.legend(handles=legend_elements, bbox_to_anchor=(-0.55, 0.0), loc='upper center', ncol=6, fontsize=18)

        plt.savefig(f'RQ2_a_c_robustness_violin_diagram.pdf', bbox_inches="tight")
        plt.savefig(f'RQ2_a_c_robustness_all_violin_diagram.svg', format='svg', bbox_inches="tight")
        plt.tight_layout()
        plt.show()

    def draw_violin_diagram_RQ3_robustness(self):

        model_list = ["gpt-3.5-turbo-1106", "llama-v2-13b-chat", "mistral-7b-instruct-4k"]
        model_name_list = ["GPT-3.5", "Llama2-13B", "Mistral-7B"]
        road_name_list = ["Road1", "Road2", "Road3", "Road4"]
        weather_name_list = ["Rainy Day", "Rainy Night", "Sunny Day", "Clear Night"]
        parameter_list = ["Position", "Rotation", "Velocity", "Angular Velocity"]
        parameter_name_list = ["Overall", "Position", "Rotation", "Velocity", "Angular Velocity"]
        color_list = [(109, 47, 32), (183, 83, 71), (223, 126, 102), (237, 199, 117), (148, 181, 148), (34, 75, 94)]
        color_list = [(r / 255.0, g / 255.0, b / 255.0) for r, g, b in color_list]
        scenario_number = 64

        csv_file_path = "./outputs_results/RQ_robustness.csv"
        data = pd.read_csv(csv_file_path)

        model_data_chart = {"Model": [], "Weather": [], "Parameter": [], "Value": []}

        for index, model in enumerate(model_list):
            scenario_index = 1
            for index_weather, weather in enumerate(weather_name_list):
                model_data = data[(data['Model'] == model) & (data['Weather'] == weather)]
                data_chart = {"Value": []}

                overall_list = []
                start_index = scenario_index
                for _ in range(4):
                    for scenario_id in range(start_index, start_index + 4):
                        scenario_data = model_data[model_data["Scenario_ID"] == scenario_id]
                        overall_list.append(scenario_data["Robustness"].mean())
                    start_index += 16
                data_chart["Value"] += overall_list
                scenario_index += 4

                for index_param, parameter in enumerate(parameter_list):
                    param_data = model_data[model_data["Parameter"] == parameter]
                    data_chart["Value"] += param_data["Robustness"].tolist()

                model_data_chart["Model"] += [model_name_list[index]] * 16 * len(parameter_name_list)
                model_data_chart["Weather"] += [weather_name_list[index_weather]] * 16 * len(parameter_name_list)
                model_data_chart["Parameter"] += np.repeat([param for param in parameter_name_list], (64 / len(weather_name_list))).tolist()
                model_data_chart["Value"] += data_chart["Value"]

        model_data_chart_df = pd.DataFrame(model_data_chart)

        sns.set(style="whitegrid", rc={'font.size': 20, 'font.weight': 'bold'})
        g = sns.FacetGrid(model_data_chart_df, row="Weather", col="Model", height=5, sharey=True)
        g.map(sns.violinplot, "Parameter", "Value",
              order=[param for param in parameter_name_list],
              palette=sns.color_palette(color_list))

        for i in range(len(weather_name_list)):
            for ax, model_title in zip(g.axes[i], model_data_chart_df['Model'].unique()):
                median_data = model_data_chart_df[(model_data_chart_df['Model'] == model_title) & (
                        model_data_chart_df['Weather'] == weather_name_list[i])].groupby('Parameter')[
                    'Value'].mean().reset_index()
                sns.pointplot(x='Parameter', y='Value', data=median_data, color='white', markers='D', join=False, ax=ax,
                              markersize=2)
                for j in range(len(parameter_name_list)):
                    ax.get_children()[3 + 4 * j].set_visible(False)

        g.set_axis_labels("", "")

        for row_ax in g.axes:
            row_ax[-1].set_ylim(-20.0, 35.0)
            row_ax[-1].set_yticks(np.arange(-20.0, 36.0, 10.0))

        for ax in g.axes.flat:
            ax.set_xticklabels([])
            ax.set_xlabel("")
            ax.set_title("")

            for label in ax.yaxis.get_ticklabels():
                label.set_fontsize(18)

            ax.spines['left'].set_color('black')
            ax.spines['left'].set_linewidth(2)
            ax.spines['bottom'].set_color('black')
            ax.spines['bottom'].set_linewidth(2)
            ax.spines['right'].set_color('black')
            ax.spines['right'].set_linewidth(2)
            ax.spines['top'].set_color('black')
            ax.spines['top'].set_linewidth(2)

        for ax, model_name in zip(g.axes[0], model_name_list):
            ax.annotate(model_name, xy=(0.5, 1.05), xycoords='axes fraction', ha='center', va='center',
                        fontsize=30, fontweight='bold')
        for ax, weather_name in zip(g.axes[:, -1], weather_name_list):
            ax.annotate(weather_name, xy=(1.05, 0.5), xycoords='axes fraction', ha='left', va='center',
                        fontsize=30, fontweight='bold', rotation=-90)

        g.axes.flat[0].annotate("Robustness Score", xy=(-0.15, -1.2), xycoords='axes fraction', ha='right', va='center',
                                fontsize=30, fontweight='bold', rotation=90)

        legend_elements = [Rectangle((0, 0), 1, 1, color=color_list[i], label=label, linewidth=0) for i, label in enumerate(parameter_name_list)]
        plt.legend(handles=legend_elements, bbox_to_anchor=(-0.55, 0.0), loc='upper center', ncol=6, fontsize=18)

        plt.savefig(f'RQ3_a_c_robustness_violin_diagram.pdf', bbox_inches="tight")
        plt.savefig(f'RQ3_a_c_robustness_violin_diagram.svg', format='svg', bbox_inches="tight")
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    parseResults = ParseResults()

    parseResults.parse_extra_mutated_scenarios()
    parseResults.count_extra_mutated_scenarios_robustness_RQ()
    parseResults.count_extra_mutated_scenarios_robustness_RQ1()
    parseResults.count_extra_mutated_scenarios_robustness_RQ2()
    parseResults.count_extra_mutated_scenarios_robustness_RQ3()

    parseResults.sort_robustness_RQ_a_c()
    parseResults.sort_rank_robustness_RQ_b_d()

    parseResults.draw_violin_diagram_RQ0_robustness()
    parseResults.draw_violin_diagram_RQ1_robustness()
    parseResults.draw_violin_diagram_RQ2_robustness()
    parseResults.draw_violin_diagram_RQ3_robustness()
