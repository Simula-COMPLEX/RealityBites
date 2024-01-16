import json
import os
import time

import fireworks.client
from openai import OpenAI


class LLMAPI(object):

    def __init__(self):
        self.total_time_step = 6

    def openai_gpt_3_5(self, model, messages, temperature):

        client = OpenAI(api_key="")  # please add your api_key here

        start_time = time.time()
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1024,
            temperature=temperature,
        )
        create_time = time.time() - start_time
        middle_time = time.time()
        output = completion.choices[0].message.content
        output_time = time.time() - middle_time
        total_time = time.time() - start_time
        return output, create_time, output_time, total_time

    def fireworks(self, model, messages, temperature):

        fireworks.client.api_key = ""  # please add your api_key here, https://app.fireworks.ai/

        def fireworks_create():
            try:
                start_time = time.time()
                completion = fireworks.client.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    max_tokens=1024,
                    temperature=temperature,
                )
                create_time = time.time() - start_time
                middle_time = time.time()
                output = completion.choices[0].message.content
                output_time = time.time() - middle_time
                total_time = time.time() - start_time
                return output, create_time, output_time, total_time
            except json.JSONDecodeError as e:
                return None, None, None, None
            except Exception as e:
                return None, None, None, None

        output, create_time, output_time, total_time = fireworks_create()
        while output is None:
            output, create_time, output_time, total_time = fireworks_create()

        return output, create_time, output_time, total_time

    def evaluate_R_MR_extra_4full_experiments(self, model, repeat_time):

        road_list = ["road1", "road2", "road3", "road4"]
        road_description = {
            "road1": "In the following scenario, Ego0's driving intention is to first drive on a dual-way road, then turn right to a one-way road of four lanes.\n",
            "road2": "In the following scenario, Ego0's driving intention is to first drive on a straight one-way road with four lanes, and then switch three lanes.\n",
            "road3": "In the following scenario, Ego0's driving intention is to first perform a left turn to switch to a straight downhill lane and then drive without turning.\n",
            "road4": "In the following scenario, Ego0's driving intention is to first turn left and then drive on the right-hand side of the road.\n"}
        weather_list = ["rain_day", "rain_night", "sunny_day", "clear_night"]
        parameter_list = ["original", "position", "rotation", "velocity", "angular_velocity"]
        extra_list = ["increase", "decrease"]

        for road in road_list:
            for weather in weather_list:
                randomly_select_scenarios_path = "./deepscenario/randomly_selected_scenarios/" + road + "-" + weather + "-scenarios/"
                randomly_select_scenarios_file_names = [f for f in os.listdir(randomly_select_scenarios_path) if
                                                        os.path.isfile(os.path.join(randomly_select_scenarios_path, f))]
                scenarios_names_list = [f for f in randomly_select_scenarios_file_names if not any(
                    param in f for param in
                    ["original", "position", "rotation", "velocity", "angular_velocity"])]
                for scenario_name in scenarios_names_list:
                    scenario_name = scenario_name.split(".deepscenario")[0]
                    randomly_select_scenario_path = randomly_select_scenarios_path + scenario_name
                    for index_parameter, parameter in enumerate(parameter_list):
                        for extra in extra_list:
                            if parameter == "original":
                                if extra == "increase":
                                    randomly_select_scenario_name = randomly_select_scenario_path + "_" + parameter + ".json"
                                else:
                                    continue
                            else:
                                randomly_select_scenario_name = randomly_select_scenario_path + "_extra_" + extra + "_" + parameter + ".json"
                            with open(randomly_select_scenario_name, 'r') as json_file:
                                randomly_select_scenario = json.load(json_file)
                            object_refs = list(randomly_select_scenario["timestep_0"].keys())
                            object_names = ""
                            for index, object_ref in enumerate(object_refs):
                                if index == len(object_refs) - 1:
                                    object_names += object_ref
                                else:
                                    object_names += object_ref + ", "

                            params_all = ""
                            for time_step in range(self.total_time_step + 1):
                                params = ""
                                for index, object_ref in enumerate(object_refs):
                                    configuration = randomly_select_scenario[f'timestep_{time_step}'][object_ref]
                                    for index_parameter_object, parameter_object in enumerate(
                                            ["position", "rotation", "velocity", "angular_velocity"]):
                                        params += f"The '{parameter_object}' of {object_ref} is ({configuration[parameter_object]['x']}, {configuration[parameter_object]['y']}, {configuration[parameter_object]['z']}).\n"
                                params_all += f"At {round(time_step * 0.5, 1)} seconds:\n{params}\n"

                            messages = []
                            sys_message = {}
                            sys_message["role"] = "system"
                            sys_message["content"] = f"""You know a lot about autonomous driving and some realistic autonomous driving crash datasets.
You are a helpful scenario realism evaluation assistant.
You will evaluate the following autonomous driving scenario, and check whether the scenario is realistic.
You will provide the corresponding realism score. The scale is 1.0-10.0, where 1.0 is unrealistic and 10.0 is realistic.
You will also provide the Probability and Confidence of the outputs as a percentage. Probability refers to the likelihood of your output, while Confidence indicates the certainty in the prediction for your output.
You know the scenario is from the LGSVL simulator."""
                            messages.append(sys_message)
                            prompt = {}
                            prompt["role"] = "user"
                            prompt[
                                "content"] = f"""{road_description[road]}The weather in this scenario is {weather.replace("_", " and ")}.
The scenario starts at 0.0 seconds and all objects start from rest.

{params_all}Your task is to perform the following actions:
1 - Evaluate the realism of the scenario for each second, and Provide the corresponding realism score.
2 - Evaluate the realism of the scenario according to each second. And Output whether the scenario is realistic, if it is realistic, output True, if not, output False. And Provide the corresponding realism score. And Output the probability and confidence for the realistic result and realism score.
3 - Output the realism scores of each second, and the realistic result and realism score of the scenario, and the corresponding probability and confidence in a JSON/json format. Here is an example:
```
{{
"0.0 seconds": <realism score>,
"0.5 seconds": <realism score>,
"1.0 seconds": <realism score>,
"1.5 seconds": <realism score>,
"2.0 seconds": <realism score>,
"2.5 seconds": <realism score>,
"3.0 seconds": <realism score>,
"realistic": <true or false>,
"realistic_probability": <probability percentage for realistic>,
"realistic_confidence": <confidence percentage for realistic>,
"scenario": <realism score>
"scenario_probability": <probability percentage for scenario realism score>,
"scenario_confidence": <confidence percentage for scenario realism score>,
}}
```

Use the following format:
Evaluation of the Realism for each second:
<evaluation results for each second>
Realism Score for each second:
<realism score for each second>
Evaluation of the Realism of the scenario:
<evaluation results>
Realistic:
<True or False>
Realistic Probability:
<probability percentage for realistic>
Realistic Confidence:
<confidence percentage for realistic>
Realism Score:
<realism score>
Realism Score Probability:
<probability percentage for scenario realism score>
Realism Score Confidence:
<confidence percentage for scenario realism score>
Realism Score in JSON/json:
<realism scores of each second, and the realism score and realistic result of the scenario, and the corresponding probability and confidence in a JSON/json format>"""
                            messages.append(prompt)

                            for index in range(repeat_time):
                                if "gpt" in model:
                                    output, create_time, output_time, total_time = self.openai_gpt_3_5(model, messages,
                                                                                                       0)
                                else:
                                    output, create_time, output_time, total_time = self.fireworks(model, messages, 0)

                                evaluate_R_MR_extra_4full_experiments_results_path = "./outputs_results/" + road + "-" + weather + "-scenarios/" + scenario_name + "/"
                                if parameter == "original":
                                    evaluate_R_MR_extra_4full_experiments_results_path += parameter + "/"
                                else:
                                    evaluate_R_MR_extra_4full_experiments_results_path += parameter + "_" + extra + "/"
                                if not os.path.exists(evaluate_R_MR_extra_4full_experiments_results_path):
                                    os.makedirs(evaluate_R_MR_extra_4full_experiments_results_path)
                                evaluate_R_MR_extra_4full_experiments_results_file_name = evaluate_R_MR_extra_4full_experiments_results_path + \
                                                                                          model.split("/")[
                                                                                              -1] + "_" + str(
                                    index) + ".txt"
                                with open(evaluate_R_MR_extra_4full_experiments_results_file_name, 'w') as file:
                                    file.write(f"model: {model}\n\n")
                                    file.write(messages[0]["content"] + "\n\n" + messages[1]["content"] + "\n\n\n")
                                    file.write(output + "\n\n\n")
                                    file.write(
                                        f"create_time: {create_time}s output_time: {output_time}s total_time: {total_time}s\n")

                                print(
                                    road + "-" + weather + " " + scenario_name + "_" + parameter + "_" + extra + "===================================================")
                                print(f"model: {model}")
                                print(f"index: {index}\n")


if __name__ == '__main__':
    llmapi = LLMAPI()

    model = "gpt-3.5-turbo-1106"
    # model = "accounts/fireworks/models/llama-v2-13b-chat"
    # model = "accounts/fireworks/models/mistral-7b-instruct-4k"

    repeat_time = 10

    llmapi.evaluate_R_MR_extra_4full_experiments(model, repeat_time)
