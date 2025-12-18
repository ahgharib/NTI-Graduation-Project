# NTI-Graduation-Project

* Study Buddy
* Lang Chain / Graph
* Multi Agents
* OCR
* Facial Expression
* YouTube and Web Search
* Explanation
* Question Making

graph TB
    %% --- Main Nodes ---
    start([User Request]):::start
    orchestrator{{Orchestrator<br/>LLM: Llama3}}:::orchestrator
    planner[Planning Agent<br/>LLM: Llama3<br/>Output: JSON roadmap]:::planner
    quiz[Quiz Agent<br/>LLM: Llama3<br/>Output: JSON quiz]:::quiz
    youtube[YouTube Agent<br/>APIs: Groq + YouTube]:::youtube
    websearch[Web Search Agent<br/>APIs: Tavily + Groq]:::websearch
    finish([Tasks Complete]):::finish
    
    %% --- Main Flow ---
    start --> orchestrator
    orchestrator -->|roadmap| planner
    orchestrator -->|quiz| quiz
    orchestrator -->|video| youtube
    orchestrator -->|search| websearch
    planner --> orchestrator
    quiz --> orchestrator
    youtube --> orchestrator
    websearch --> orchestrator
    orchestrator -->|done| finish
    
    %% --- Tools Subgraph ---
    subgraph tools[Internal Tools]
        plantools[PlanTools]:::tool
        validation[ValidationTools]:::tool
        orchestration[OrchestrationTools]:::tool
    end
    
    %% --- APIs Subgraph ---
    subgraph apis[External APIs]
        groq[Groq API]:::api
        youtube_api[YouTube API]:::api
        tavily[Tavily Search]:::api
    end
    
    %% --- Tool Connections ---
    planner -.-> plantools
    orchestrator -.-> validation
    orchestrator -.-> orchestration
    youtube -.-> groq
    youtube -.-> youtube_api
    websearch -.-> tavily
    websearch -.-> groq
    
    %% --- Styling ---
    classDef orchestrator fill:#FFF3E0,stroke:#FF9800,stroke-width:3px
    classDef planner fill:#E8F5E8,stroke:#4CAF50,stroke-width:2px
    classDef quiz fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    classDef youtube fill:#E3F2FD,stroke:#2196F3,stroke-width:2px
    classDef websearch fill:#FFF8E1,stroke:#FFC107,stroke-width:2px
    classDef start fill:#E8EAF6,stroke:#3F51B5,stroke-width:2px
    classDef finish fill:#FCE4EC,stroke:#E91E63,stroke-width:2px
    classDef tool fill:#E0F2F1,stroke:#009688,stroke-width:2px
    classDef api fill:#FFEBEE,stroke:#F44336,stroke-width:2px
    
