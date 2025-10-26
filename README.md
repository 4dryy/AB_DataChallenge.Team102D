# AB Data Challenge – Team 102.D

**Course:** Project Management  
**Team:** 102.D  
**Project:** Detecció de Consums Anòmals (Aigües de Barcelona Data Challenge)

## Project Description

This project focuses on anomaly detection in water consumption data for Aigües de Barcelona. The goal is to develop a robust system capable of identifying unusual consumption patterns that may indicate leaks, meter malfunctions, or other anomalies in the water distribution network.

## Context

We continue with the project even if not selected for the official challenge. We will use the provided `data/dataset_sample.parquet` file or implement a synthetic data fallback to ensure project continuity and learning objectives.

## Objectives / Success Criteria

- **Timely Delivery:** Complete Iteration-1 within the specified timeframe
- **Quality Documentation:** Maintain comprehensive documentation throughout the project
- **Anomaly Detection Performance:** Achieve ≥90% recall with <10% false positive rate (placeholder target)
- **Technical Excellence:** Deliver clean, maintainable code with proper validation

## Constraints / Assumptions

- **Budget:** No budget allocated for external tools or services
- **Data Access:** Limited to sample dataset provided
- **Scope:** Prototype-level implementation for learning and demonstration purposes
- **Timeline:** Iteration-based development approach

## Iteration-1 Pipeline (WBS-aligned)

### 1. Data Understanding & Acquisition
- Load and validate the provided dataset (`data/dataset_sample.csv`)
- Analyze data quality and characteristics
- Summarize data characteristics by policy
- Identify data quality issues and anomalies

### 2. Feature Definition & Exploration Plan
- Define candidate feature families for anomaly detection
- Create comprehensive EDA (Exploratory Data Analysis) plan
- Establish data cleaning rules and preprocessing requirements
- Plan feature engineering approach for Iteration 2

## Stakeholders

- **Team 102.D:** Primary development team
- **Course Instructors:** Project guidance and evaluation
- **Aigües de Barcelona:** Data provider and potential end-user
- **Academic Community:** Knowledge sharing and peer review

## Risk Snapshot

| Risk | Impact | Probability | Strategy |
|------|--------|-------------|----------|
| Data Quality Issues | High | Medium | Implement robust validation and synthetic fallback |
| Limited Dataset Size | Medium | High | Use synthetic data generation and feature engineering |
| Technical Complexity | Medium | Medium | Iterative development with clear documentation |
| Timeline Constraints | High | Low | Agile approach with clear milestones |
| Performance Requirements | Medium | Medium | Start with baseline and iterate for optimization |

## How to Run

1. Open both notebooks in the `iteration_1/` directory:
   - `01_data_understanding.ipynb` - Data loading, validation, and initial analysis
   - `02_feature_plan.ipynb` - Feature definition and EDA planning

2. Execute the notebooks in order to follow the Iteration-1 pipeline

3. Review the conclusions and next steps outlined in each notebook

## Repository Structure

```
AB_DataChallenge.Team102D/
├── data/                               # Data storage
│   └── dataset_sample.parquet          # Sample dataset for analysis
├── docs/                               # Project documentation
│   ├── Team102D.ProjectCharter.v0.0.pdf
│   ├── Team102D.ProjectStakeholderMatrix.v.0.1 - Stakeholder Matrix.pdf
│   ├── Team102D.ProjectWorkPlan.v1.4.docx.pdf
│   └── Team102D.Follow-upRegister.v0.1 - Risk.pdf
├── iteration_1/                        # Iteration 1 implementation
│   ├── 01_data_understanding.ipynb    # Data analysis and validation
│   └── 02_feature_plan.ipynb          # Feature planning and EDA
├── iteration_2/                        # Iteration 2 implementation (future)
│   └── (to be implemented)
├── iteration_3/                        # Iteration 3 implementation (future)
│   └── (to be implemented)
├── models/                             # Model storage (future)
│   └── (to be implemented)
├── results/                            # Results and outputs (future)
│   └── (to be implemented)
├── .gitignore                          # Git ignore rules
└── README.md                           # This file
```

### Directory Descriptions

- **`data/`**: Contains all datasets used in the project
- **`docs/`**: Project documentation including charters, stakeholder matrices, work plans, and risk registers
- **`iteration_1/`**: Complete implementation of Iteration 1 (Data Understanding & Feature Planning)
- **`iteration_2/`**: Future implementation of Iteration 2 (Feature Engineering & Model Development)
- **`iteration_3/`**: Future implementation of Iteration 3 (Model Optimization & Deployment)
- **`models/`**: Future storage for trained models and model artifacts
- **`results/`**: Future storage for analysis results, visualizations, and reports