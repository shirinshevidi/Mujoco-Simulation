#!/usr/bin/env python3

import os
from openai import OpenAI  # Add this import
from pydantic import Field
import json
import logging
import time
from typing import List
import instructor
from instructor import OpenAISchema

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RoboticAction(OpenAISchema):
    """
    Represents a single robotic action in the disassembly sequence.
    """
    human_working: bool = Field(..., description="Indicates if a human is working alongside the robot")
    selected_element: str = Field(..., description="The element being worked on")
    planning_sequence: List[str] = Field(..., description="List of actions for the robot to perform")

class ActionSequence(OpenAISchema):
    """
    Represents a sequence of robotic actions for the disassembly plan.
    """
    actions: List[RoboticAction] = Field(..., description="List of robotic actions to perform")

class PlanningAgent:
    def __init__(self):
        # Initialize OpenAI client with Instructor
        self.client = instructor.patch(OpenAI())

        # Dictionary of possible robot actions
        self.robot_actions = {
            "move_in_cartesian_path": "move_in_cartesian_path(move_distance_x, move_distance_y, move_distance_z)",
            "moveto": "moveto",
            "picking": "picking",
            "holding": "holding",
            "placing": "placing",
            "human_action": "human_action(action_description)"
        }

        logging.info("Planning Agent Initialized and ready to work.")

    def handle_plan_execution(self, plan):
        logging.info(f"Planning Agent: Received plan execution request: {plan}")

        # Translate plan into structured action sequences
        action_sequence = self.translate_plan(plan)

        if action_sequence is None:
            return False, "Failed to generate a valid action sequence."

        # Validate action sequence
        if self.validate_action_sequence(action_sequence):
            # Execute preliminary steps
            self.execute_preliminary_steps()

            # Check if additional safety measures are needed
            if "unsafe" in plan.lower() or "modifications" in plan.lower():
                action_sequence = self.add_safety_measures(action_sequence)

            # Write the JSON file
            json_file_path = self.write_json_file(action_sequence)

            # Execute actions
            success, execution_details = self.execute_actions(action_sequence)
            return success, f"{execution_details}. JSON file created at {json_file_path}"
        else:
            return False, "Invalid action sequence. Please check the plan and try again."

    def add_safety_measures(self, action_sequence):
        logging.info("Planning Agent: Adding additional safety measures to the action sequence")
        safety_measures = [
            "implement_temporary_supports",
            "distribute_load_evenly",
            "monitor_stability_continuously"
        ]
        action_sequence["planning_sequence"] = safety_measures + action_sequence["planning_sequence"]
        return action_sequence

    def translate_plan(self, plan):
        logging.info(f"Planning Agent: Translating plan: {plan}")
        action_schemas = ', '.join(f"{key}: {value}" for key, value in self.robot_actions.items())
        prompt = f"""
        You are the Planning Agent in a multi-agent system that controls a robotic arm for disassembly tasks. Your role is to translate the disassembly sequence plan into a structured action sequence for the robotic arm to execute, collaborating with a human operator. 

        Given the disassembly plan:
        {plan}

        Your task is to:
        1. Analyze the given disassembly plan, focusing primarily on the numbered Disassembly Instructions.
        2. Create a detailed action sequence using only the following action schemas:
           {action_schemas}
        3. Ensure the action sequence follows the EXACT order specified in the numbered Disassembly Instructions.
        4. Maintain consistent roles for each actor throughout the entire process as defined in the numbered instructions.
        5. Identify the specific element being worked on in each step.
        6. Ensure that if an actor is instructed to support an element, they continue to do so until explicitly instructed to release it.

        Guidelines:
        - Prioritize the numbered Disassembly Instructions over any additional comments or information provided.
        - Follow the disassembly instructions step by step, without changing the order or assigned roles.
        - Break down complex movements into a series of simpler actions.
        - Include necessary preparatory movements before each main action.
        - For human actions, use the format: human_action(action_description)
        - Use specific element names (e.g., "element_1" instead of "element 1") for consistency.
        - Use EXACTLY the action names provided (e.g., 'moveto' not 'move_to').
        - Set human_working to true for steps performed by humans, and false for steps performed by the robot.
        - When human_working is true, only include human_action in the planning_sequence.
        - When human_working is false, only include robot actions in the planning_sequence.
        - Ensure that each actor maintains their assigned role throughout the entire process as specified in the numbered instructions.

        Ensure that:
        1. "human_working" is set appropriately based on whether the action is performed by a human or the robot, as specified in the numbered instructions.
        2. "selected_element" specifies the element being worked on in the current step.
        3. The actions in the "planning_sequence" are organized in execution order.
        4. Robot pick-and-place sequences follow this pattern: moveto -> picking -> holding -> placing
        5. Support actions follow this pattern: moveto -> holding, and continue holding in subsequent steps
        6. Use 'deposition_zone' as the destination for removed elements.
        7. Each actor maintains their assigned role consistently throughout the entire sequence as per the numbered instructions.

        Note: Your response will be automatically parsed into a structure like this:
        {{
            "actions": [
                {{
                    "human_working": boolean,
                    "selected_element": "element_name",
                    "planning_sequence": ["action1", "action2", ...]
                }},
                ...
            ]
        }}

        Planning Agent, please provide the structured action sequence based on the given disassembly plan, ensuring clear collaboration between actors, maintaining continuous support of elements as specified, and keeping each actor in their assigned role throughout the entire process as defined in the numbered Disassembly Instructions.
        Re-evaluate your plan and make sure it matches the numbered disassembly sequence plan exactly, maintaining consistent roles for each actor as specified in those instructions.
        """
        try:
            action_sequence = self.client.chat.completions.create(
                model="gpt-4o",
                response_model=ActionSequence,
                temperature=0,  # Add this line to set the temperature (value between 0 and 2)
                messages=[
                    {"role": "system", "content": "You are a planning agent that translates disassembly plans into structured action sequences."},
                    {"role": "user", "content": prompt}
                ]
            )
            logging.info(f"Planning Agent: Translated plan into action sequence: {action_sequence}")
            return action_sequence
        except Exception as e:
            logging.error(f"Planning Agent: Failed to generate action sequence: {e}")
            return None

    def validate_action_sequence(self, action_sequence):
        if not isinstance(action_sequence, ActionSequence):
            logging.error("Planning Agent: Action sequence is not an ActionSequence object")
            return False

        for item in action_sequence.actions:
            if not isinstance(item, RoboticAction):
                logging.error(f"Planning Agent: Invalid item in action sequence: {item}")
                return False

            # Check if all actions in the sequence are valid
            invalid_actions = []
            for action in item.planning_sequence:
                action_name = action.split('(')[0]  # Extract action name
                if not any(action_name == valid for valid in self.robot_actions.keys()):
                    invalid_actions.append(action)

            if invalid_actions:
                logging.error(f"Planning Agent: Invalid actions in sequence: {', '.join(invalid_actions)}")
                return False

        logging.info("Planning Agent: Action sequence validated successfully")
        return True

    def execute_preliminary_steps(self):
        # Placeholder for executing preliminary steps
        logging.info("Planning Agent: Executing preliminary steps for safety.")

    def write_json_file(self, action_sequence):
        # Write the action sequence to a JSON file
        robot_sequence_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'robot_sequence')
        os.makedirs(robot_sequence_dir, exist_ok=True)
        timestamp = time.time()
        json_file_name = f"action_sequence_{timestamp}.json"
        json_file_path = os.path.join(robot_sequence_dir, json_file_name)
        
        with open(json_file_path, 'w') as json_file:
            json.dump(action_sequence.dict(), json_file, indent=4)
        logging.info(f"Planning Agent: Action sequence JSON file created at {json_file_path}")
        
        # Print the contents of the JSON file
        with open(json_file_path, 'r') as json_file:
            logging.info(f"Planning Agent: JSON file contents:\n{json_file.read()}")
        
        return json_file_path

    def execute_actions(self, action_sequence):
        # Simulate executing actions
        try:
            for action in action_sequence.actions:
                for step in action.planning_sequence:
                    logging.info(f"Planning Agent: Executed action: {step}")
            return True, "Action sequence executed successfully."
        except Exception as e:
            logging.error(f"Planning Agent: Error executing actions: {e}")
            return False, f"Error executing actions: {e}"

def main():
    planning_agent = PlanningAgent()
    plan = """
    Description of the Structure:
    - Simple portal frame structure with three elements: two vertical columns (column 1 and column 3) and one horizontal beam (beam 2).
    - The structure forms a basic 'П' shape, and all elements are assumed to be of standard construction material (like timber, steel, or concrete) with the same dimensions.

    Disassembly Instructions:
    1. actor_1 supports the beam (element 2) to secure the structure.
    2. actor_2 removes the vertical column (1) from below the beam (2).
    3. actor_2 removes the vertical column (3) from below the beam (2).
    4. Finally, actor_1 carefully removes the beam (2) that is being supported last and places it in the deposition zone.
    
    -I, the human, will remove column 1.
    """
    success, details = planning_agent.handle_plan_execution(plan)
    if success:
        logging.info(f"Plan executed successfully: {details}")
    else:
        logging.error(f"Plan execution failed: {details}")

if __name__ == "__main__":
    main()