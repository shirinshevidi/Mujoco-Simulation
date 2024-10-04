## Project Overview

This project implements an advanced Intelligent Robotic Disassembly System, leveraging a multi-agent architecture to automate and optimize the disassembly process of complex structures. The system is designed to interpret natural language commands, validate disassembly procedures, ensure structural stability, and execute precise robotic actions sequence as JSON files.


## Agent description

### **1. Manager Agent (Coordinator and Interpreter)**

- **Function**: Acts as the central hub, coordinating all interactions and assigning tasks to other agents. It interprets the user's natural language commands, even if they are ambiguous or complex, and maintains context over multiple interactions to ensure seamless communication.
- **Key Enhancements**:
    - **Robust Natural Language Understanding**: Equipped with advanced NLU capabilities to handle ambiguous or complex user commands effectively.
    - **Context-Aware Processing**: Utilizes `ConversationBufferMemory` to maintain conversation history and context.
    - **Clear Prompt Guidelines**: Employs well-defined prompts to handle inputs gracefully and reduce misunderstandings.
- **Key Tasks**:
    - Accurately interprets the user's commands.
    - Maintains context for continuity across interactions.
    - Routes tasks to the **Structural Engineer Agent** for validation.
    - Coordinates feedback from other agents.
    - Communicates results back to the user promptly.
- **Pipeline Steps**:
    1. Receives the user's input.
    2. Interprets the intent, handling any ambiguities.
    3. Maintains context using conversation memory.
    4. Routes the task to the **Structural Engineer Agent** for validation.
    5. Receives results from other agents and communicates them to the user.

---

### **2. Structural Engineer Agent (Initial Validator)**

- **Function**: Serves as the first point of validation, ensuring that the user's request aligns with standard disassembly procedures. It accesses an up-to-date **Retrieval-Augmented Generation (RAG)** system containing the latest disassembly manuals.
- **Key Enhancements**:
    - **Updated RAG System**: Regularly updates the vector store with the latest manuals and procedures for accuracy.
    - **Feedback Mechanism**: Implements a system to learn from new situations not covered in the manuals. Via stability and successful actions
    - **(might be implemented later) Domain-Specific Embeddings**: Utilizes embeddings tailored to the domain for better retrieval accuracy.
- **Key Tasks**:
    - Validates the user's request against the most current disassembly manuals.
    - Approves standard requests and communicates back to the **Manager Agent**.
    - For non-standard or risky requests, forwards the task to the **Stability Agent**.
- **Pipeline Steps**:
    1. Receives the request from the **Manager Agent**.
    2. Queries the updated RAG system using domain-specific embeddings.
    3. Validates the request.
        - **If Non-Standard**: Forwards to the **Stability Agent** for further analysis.
        - **If Standard**: Approves and sends back to the **Manager Agent**.

---

### **3. Stability Agent (Structural Safety Analysis)**

- **Function**: Ensures that the requested action won't compromise structural stability. It evaluates the impact of non-standard or complex actions and suggests additional steps if necessary. The agent leverages advanced predictive capabilities, including physics simulations and machine learning models.
- **Key Enhancements**:
    - **Enhanced Predictive Capabilities**: Integrates physics simulation models and real-time data for accurate analysis.
    - **Learning from Past Assessments**: Uses feedback mechanisms to enhance future evaluations. Successful task completions will be added to the RAG
    - **External Simulation Tools**: simulation in unity
- **Key Tasks**:
    - Receives non-standard tasks from the **Structural Engineer Agent**.
    - Evaluates structural risks using simulations and ML models.
    - Suggests additional steps to maintain stability.
    - Communicates the modified plan to the **Planning Agent**.
- **Pipeline Steps**:
    1. Receives the non-standard task.
    2. Analyzes risks using simulations and ML models.
    3. Suggests necessary adjustments (e.g., supporting elements).
    4. Passes the modifications to the **Planning Agent**.

---

### **4. Planning Agent (Execution Planner)**

- **Function**: Translates validated or modified plans into specific, low-level robotic actions. Ensures that all actions are safe and align with the intended high-level plan by using standardized schemas and error-checking mechanisms.
- **Key Enhancements**:
    - **Standardized Action Schemas**: Utilizes detailed library of actions to standardize the translation process.
    - **Error-Checking Mechanisms**: Verifies that low-level actions match the high-level plan.
    - **Structured Output Formats**: Generates action sequences in clear JSON format. To guarantee a structure of the JSON, we can use instructor library: https://python.useinstructor.com/examples/
    - **Validation Against Schemas**: Ensures actions are compliant with predefined structures to prevent errors.
- **Key Tasks**:
    - Translates plans into detailed robotic action sequences.
    - Validates and error-checks action sequences.
    - Creates a JSON file that can be used to control the robot. The structure of this JSON looks like this:
    - The structure of this JSON looks like this:
    ```json
    {
      "human_working": true,
      "selected_element": "element3",
      "planning_sequence": [
        "moveto(home)",
        "holding",
        "picking",
        "moveto(home)"
      ]
    }
    ```
    - Reports execution status back to the **Manager Agent**.
- **Pipeline Steps**:
    1. Receives the plan from the **Stability Agent** or **Structural Engineer Agent**.
    2. Translates the plan into structured action sequences.
    3. Validates actions against schemas with error-checking.
    4. Writes the JSON file.
    5. Reports back to the **Manager Agent**.


## High-Level LangChain Setup for Multi-Agent Robotic Control System

### **1. Overall Structure**

- **Components**:
    - **LLMChains**: Customized for each agent with tailored prompts.
    - **Tools**: For specific functionalities like querying the RAG system and running simulations.
    - **AgentExecutor**: Manages complex behaviors and interactions between agents.
    - **Vector Stores**: Implements RAG using systems like FAISS with domain-specific embeddings.
    - **Memory Management**: Uses `ConversationBufferMemory` for context management across interactions.
- **Performance Optimization**:
    - Optimize each component to prevent bottlenecks.
    - Implement caching strategies where appropriate to reduce latency.

### **2. Individual Agent Setup**

### **Manager Agent**

- **LLMChain**:
    - Custom prompt defining its coordinating role.
    - Incorporates clear guidelines to handle ambiguous inputs.
- **Memory**:
    - Utilizes `ConversationBufferMemory` to maintain context.
- **Language Model**:
    - Employs models like `ChatOpenAI`

### **Structural Engineer Agent**

- **RAG System**:
    - **Vector Store**: Regularly updated with the latest manuals.
    - **Embeddings**: Domain-specific for accurate retrieval.
- **Tool**:
    - Custom tool for efficient querying of the vector store.
- **LLMChain**:
    - Prompt instructs validation against manuals.
- **AgentExecutor**:
    - Combines LLMChain and RAG Tool for seamless operation.
- **Feedback Mechanism**:
    - Learns from new scenarios to enhance future validations.

### **Stability Agent**

- **LLMChain**:
    - Specialized prompt focused on structural analysis.
- **Tools**:
    - **External Simulation Tools**: Integrated for advanced analysis.
- **Feedback Mechanism**:
    - Learns from past assessments to refine future analyses.

### **Planning Agent**

- **LLMChain**:
    - Prompt guides translation of plans into sequence planning for robotic actions.
- **Output Formats**:
    - Generates action sequences in structured format of JSON.
- **Validation**:
    - Uses predefined schemas to validate action sequences.
- **Error-Checking**:
    - Incorporates mechanisms to ensure actions align with plans.

### **3. System Integration**

- **Orchestrator**:
    - Central function or class managing agent interactions.
    - Implements concurrency to enhance responsiveness.
    - Uses a message-passing interface or event-driven architecture for scalability.
- **Error Handling and Checkpoints**:
    - Adds checkpoints after major steps for exception handling and retries.
    - Defines specific error types and procedures for different failure modes.
- **Logging and Monitoring**:
    - Implements comprehensive logging for performance tracking and diagnostics.
- **Security and Safety**:
    - Incorporates authentication and authorization mechanisms.
    - Utilizes sandboxing when executing code generated by LLMs.

### **4. Workflow**

1. **User Input → Manager Agent**:
    - Interprets the command with advanced NLU capabilities.
    - Maintains context using conversation memory.
2. **Manager Agent → Structural Engineer Agent**:
    - Validates the request against manuals using the updated RAG system.
3. **Structural Engineer Agent → Stability Agent (if needed)**:
    - Analyzes non-standard requests for safety using simulations and ML models.
4. **Stability Agent/Structural Engineer Agent → Planning Agent**:
    - Translates validated plans into structured robotic actions.
5. **Planning Agent → Robot Control Interface**:
    - Executes the action sequence after validation and error-checking.
6. **Feedback Loop**:
    - Results and execution statuses are reported back through the agents.
    - Agents update their knowledge bases based on outcomes.

### **5. Key Considerations**

- **Error Handling**:
    - Implements robust error types and handling procedures.
    - Includes fallback mechanisms for uninterrupted operation.
- **Safety and Ethics Compliance**:
    - Adheres to safety protocols in autonomous operations.
    - Ensures ethical standards are maintained throughout processes.
- **Performance Optimization**:
    - Caches frequent queries and responses to enhance efficiency.
    - Profiles the system to identify and optimize slow components.
- **Testing and Validation**:
    - Conducts unit tests for individual agents.
    - Performs integration tests for the entire system.
    - Uses simulation environments before deploying actions on physical hardware.
- **Documentation and Maintenance**:
    - Maintains comprehensive documentation of code and prompts.
    - Establishes a maintenance plan for regular updates.


## Technologies Used

- ROS (Robot Operating System)
- Python
- Natural Language Processing
- Retrieval-Augmented Generation (RAG) Systems