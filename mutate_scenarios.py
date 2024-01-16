import json
import math
import os
import random

import xml.etree.ElementTree as ET

import lgsvl


class MutateScenarios(object):

    def __init__(self):
        self.total_time_step = 6

    def _convert_to_float(self, configuration):
        if isinstance(configuration, dict):
            return {key: self._convert_to_float(value) for key, value in configuration.items()}
        elif isinstance(configuration, str):
            return float(configuration)
        else:
            return configuration

    def _get_all_objects_names(self, xml_content):
        root = ET.fromstring(xml_content)
        object_refs = []
        for object_init in root.findall('.//ObjectInitialization'):
            object_ref = object_init.attrib['objectRef']
            object_refs.append(object_ref)
        return object_refs

    def original_scenarios_2json(self):

        road_list = ["road1", "road2", "road3", "road4"]
        weather_list = ["rain_day", "rain_night", "sunny_day", "clear_night"]
        runner = lgsvl.scenariotoolset.ScenarioRunner()

        for road in road_list:
            for weather in weather_list:
                randomly_select_scenarios_path = "./deepscenario/randomly_selected_scenarios/" + road + "-" + weather + "-scenarios/"
                randomly_select_scenarios_names_list = [f for f in os.listdir(randomly_select_scenarios_path) if
                                                        os.path.isfile(os.path.join(randomly_select_scenarios_path, f))]
                for randomly_select_scenario_name in randomly_select_scenarios_names_list:
                    randomly_select_scenario_name = os.path.join(randomly_select_scenarios_path,
                                                                 randomly_select_scenario_name)
                    runner.load_scenario_file(scenario_filepath_or_buffer=randomly_select_scenario_name)
                    with open(randomly_select_scenario_name, 'r', encoding='utf-8') as file:
                        randomly_select_scenario_xml_content = file.read()

                    object_refs = self._get_all_objects_names(randomly_select_scenario_xml_content)
                    parameters_4timestep = {}
                    for time_step in range(1, self.total_time_step + 2):
                        info = json.loads(runner.get_scene_by_timestep(timestep=time_step))
                        parameters_4object = {}
                        for index_object, object_ref in enumerate(object_refs):
                            parameters_4final = {}
                            for index_parameter_object, parameter_object in enumerate(
                                    ["position", "rotation", "velocity", "angular_velocity"]):
                                if parameter_object == "position" or parameter_object == "rotation":
                                    params_type = "geographic_parameters"
                                else:
                                    params_type = "dynamic_parameters"
                                parameter_4final = self._convert_to_float(
                                    info[object_ref][params_type][parameter_object])
                                parameters_4final[parameter_object] = parameter_4final
                            parameters_4object[object_ref] = parameters_4final
                        parameters_4timestep[f"timestep_{time_step - 1}"] = parameters_4object
                    with open(f"{randomly_select_scenario_name.split('.deepscenario')[0]}_original.json",
                              'w') as json_file:
                        json.dump(parameters_4timestep, json_file, indent=2)

    def _extra_mutation(self, parameter, mutation_per=0.01):
        return round(parameter * (1 + mutation_per), 3)

    def extra_mutate_scenarios_7timestep(self):

        road_list = ["road1", "road2", "road3", "road4"]
        weather_list = ["rain_day", "rain_night", "sunny_day", "clear_night"]
        parameter_list = ["position", "rotation", "velocity", "angular_velocity"]
        object_list = ["Ego0", "NPC0"]
        mutation_per_list = [0.01, -0.01]
        mutation_dimension = False

        for mutation_per in mutation_per_list:
            for road in road_list:
                for weather in weather_list:
                    randomly_select_scenarios_path = "./deepscenario/randomly_selected_scenarios/" + road + "-" + weather + "-scenarios/"
                    randomly_select_scenarios_names_list = [f for f in os.listdir(randomly_select_scenarios_path) if
                                                            os.path.isfile(os.path.join(randomly_select_scenarios_path,
                                                                                        f)) and "original.json" in f]
                    for randomly_select_scenario_name in randomly_select_scenarios_names_list:
                        with open(randomly_select_scenarios_path + randomly_select_scenario_name, 'r') as file:
                            randomly_select_scenario = json.load(file)
                        for parameter in parameter_list:
                            parameters_4timestep = {}
                            for time_step in range(1, self.total_time_step + 2):
                                parameters_4object = {}
                                for object in object_list:
                                    parameters_4final = {}
                                    if mutation_dimension:
                                        index_parameter = random.choice(["x", "y", "z"])
                                        for parameter_object in ["position", "rotation", "velocity",
                                                                 "angular_velocity"]:
                                            parameter_4final = self._convert_to_float(
                                                randomly_select_scenario[f"timestep_{time_step - 1}"][object][
                                                    parameter_object])
                                            if parameter == parameter_object:
                                                mutated_parameter = self._extra_mutation(
                                                    parameter_4final[index_parameter], mutation_per)
                                                parameter_4final[index_parameter] = mutated_parameter
                                            parameters_4final[parameter_object] = parameter_4final
                                    else:
                                        for parameter_object in ["position", "rotation", "velocity",
                                                                 "angular_velocity"]:
                                            parameter_4final = self._convert_to_float(
                                                randomly_select_scenario[f"timestep_{time_step - 1}"][object][
                                                    parameter_object])
                                            if parameter == parameter_object and object == "NPC0":
                                                for index_parameter in ["x", "y", "z"]:
                                                    mutated_parameter = self._extra_mutation(
                                                        parameter_4final[index_parameter], mutation_per)
                                                    parameter_4final[index_parameter] = mutated_parameter
                                            parameters_4final[parameter_object] = parameter_4final
                                    parameters_4object[object] = parameters_4final
                                parameters_4timestep[f"timestep_{time_step - 1}"] = parameters_4object
                            with open(
                                    f"{randomly_select_scenarios_path}{randomly_select_scenario_name.split('_original.json')[0]}_extra_{'increase' if mutation_per == 0.01 else 'decrease'}_{parameter}.json",
                                    'w') as json_file:
                                json.dump(parameters_4timestep, json_file, indent=2)

                            if parameter == "original":
                                continue
                            tree = ET.parse(
                                randomly_select_scenarios_path + randomly_select_scenario_name.split('_original.json')[
                                    0] + ".deepscenario")
                            root = tree.getroot()
                            for time_step in range(1, self.total_time_step + 2):
                                for object_action in root.iter('ObjectAction'):
                                    object_ref = object_action.get('name').split('_')[1]
                                    object_parameters = parameters_4timestep[f'timestep_{time_step - 1}'][object_ref]
                                    for waypoint in object_action.iter('WayPoint'):
                                        timestamp = waypoint.get('timeStamp')
                                        if int(timestamp) == time_step:
                                            for index_parameter_object, parameter_object in enumerate(
                                                    ["position", "rotation", "velocity",
                                                     "angular_velocity"]):
                                                if parameter_object == "position" or parameter_object == "rotation":
                                                    parameter_type = 'GeographicParameters/Object'
                                                else:
                                                    parameter_type = 'DynamicParameters/'
                                                object_parameter = waypoint.find(
                                                    f'{parameter_type}{"AngularVelocity" if parameter_object == "angular_velocity" else parameter_object.capitalize()}')
                                                if object_parameter is not None:
                                                    if parameter_object == "velocity":
                                                        speed = round(math.sqrt(
                                                            object_parameters[parameter_object]['x'] ** 2 +
                                                            object_parameters[parameter_object]['y'] ** 2 +
                                                            object_parameters[parameter_object]['z'] ** 2), 3)
                                                        waypoint.find(f'{parameter_type}Speed').set('speed', str(speed))
                                                    object_parameter.set(
                                                        f'{"angularVelocity" if parameter_object == "angular_velocity" else parameter_object}X',
                                                        str(object_parameters[parameter_object]['x']))
                                                    object_parameter.set(
                                                        f'{"angularVelocity" if parameter_object == "angular_velocity" else parameter_object}Y',
                                                        str(object_parameters[parameter_object]['y']))
                                                    object_parameter.set(
                                                        f'{"angularVelocity" if parameter_object == "angular_velocity" else parameter_object}Z',
                                                        str(object_parameters[parameter_object]['z']))
                            save_file = f"{randomly_select_scenarios_path}{randomly_select_scenario_name.split('_original.json')[0]}_extra_{'increase' if mutation_per == 0.01 else 'decrease'}_{parameter}.deepscenario"
                            tree.write(save_file)
                            with open(save_file, 'r') as file:
                                xml_content = file.read()
                                xml_content = '<?xml version="1.0" encoding="utf-8"?>\n' + xml_content
                            with open(save_file, 'w') as file:
                                file.write(xml_content)


if __name__ == '__main__':
    mutate_scenarios = MutateScenarios()
    mutate_scenarios.original_scenarios_2json()
    mutate_scenarios.extra_mutate_scenarios_7timestep()
