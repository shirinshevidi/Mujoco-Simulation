# Engineering Conventions

## Conventions
> When writing code, follow these conventions.

- Write simple, verbose code over terse, compact, dense code.
- If a function does not have a coresponding test, mention it.
- When building test, don't mock anything.

## Project Structure

> The project follows a ROS (Robot Operating System) catkin workspace structure. Here's an overview of the project structure:

src/
├── CMakeLists.txt
├── multi_agent_system/
│ └── package.xml
├── planning_agent/
│ └── package.xml
├── stability_agent/
│ ├── CMakeLists.txt
│ └── package.xml
└── structural_engineer_agent/
├── CMakeLists.txt
└── package.xml
data/
└── disassembly_manuals.json

