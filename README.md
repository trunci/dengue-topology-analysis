# Dengue Transmission Network Analysis

Authors: Eduardo Trunci, João Trunci.

A sophisticated analysis tool that applies **Discrete Morse Theory** to identify critical transmission hubs in Brazil's dengue network using municipality-level data from SINAN (Sistema de Informação de Agravos de Notificação).

## Overview

This project analyzes dengue transmission patterns across Brazilian municipalities to identify:

- **Critical transmission hubs** that serve as bridges between regions
- **Outbreak sources** where epidemics often originate
- **High-risk municipalities** requiring prioritized surveillance
- **Transmission pathways** based on imported cases and temporal correlations

## Methodology

### Discrete Morse Theory Application

The analysis applies discrete Morse theory to identify critical points in the transmission network:

- **Local Maxima**: High-risk outbreak sources
- **Saddle Points**: Critical transmission bridges connecting different regions
- **Local Minima**: Low-risk areas

### Network Construction

The transmission network is built using:

1. **Imported Cases**: Direct transmission pathways from infection location to residence
2. **Temporal Correlations**: Municipalities with correlated outbreak patterns
3. **Risk Scoring**: Multi-component scoring system considering case counts, severity, importation rates, and trends

## Quick Start

### Prereqs

```bash
pip install pandas numpy networkx
```

### Running the Analysis

```bash
python main.py
```

### Configuration

Edit the `DATA_SOURCE` variable in `main.py` to point to your CSV file:

```python
DATA_SOURCE = "./DENGBR25.csv"  # Change this path
```

## Structure

```
deng/
├── main.py                    # Main analysis script
├── DENGBR25.csv              # SINAN dengue data (1.5M+ records)
├── dengue_analysis.log       # Detailed analysis log
├── transmission_hubs.csv     # Results output
└── README.md                 # This file
```

## Data Requirements

The analysis expects a CSV file with the following key columns:

- `id_mn_resi`: Municipality of residence (required)
- `comuninf`: Municipality of probable infection
- `tpautocto`: Autochthony status (1=local, 2=imported, 3=indeterminate)
- `classi_fin`: Case classification (1=confirmed, 2=discarded)
- `dt_sin_pri`: Date of first symptoms
- `sem_pri`: Epidemiological week
- `hospitaliz`: Hospitalization status
- `evolucao`: Case outcome

## Analysis Pipeline

The script follows a 5-step analysis pipeline:

1. **Data Loading & Preprocessing**

   - Loads SINAN data with 121 columns
   - Filters confirmed cases
   - Handles missing municipality data
   - Converts date fields

2. **Network Construction**

   - Creates directed graph with municipalities as nodes
   - Adds edges from imported cases (actual transmission)
   - Computes temporal correlations between municipalities
   - Builds comprehensive transmission network

3. **Risk Score Computation**

   - Multi-component scoring system:
     - Case count (log-transformed)
     - Severity rate (hospitalizations, deaths)
     - Importation rate
     - Recent trend analysis

4. **Discrete Morse Theory Application**

   - Computes discrete gradient vector field
   - Identifies critical points (unpaired nodes)
   - Classifies as maxima, saddles, or minima

5. **Hub Identification**
   - Combines Morse theory results with network centrality
   - Ranks municipalities by transmission importance
   - Generates comprehensive report

## Output Files

### `transmission_hubs.csv`

Contains the top 20 transmission hubs with:

- Municipality ID
- Hub score
- Risk score
- Morse theory classification
- Network degree

### `dengue_analysis.log`

Detailed analysis log including:

- Data quality statistics
- Network construction progress
- Critical point identification
- Hub ranking and characteristics

## Key Insights

1. **Bridge Municipalities**: Small municipalities with few cases but high connectivity serve as critical transmission bridges
2. **Major Urban Centers**: Large cities like São Paulo (355030.0) and Rio de Janeiro (330455.0) are major connectors
3. **Geographic Patterns**: Critical hubs are distributed across different Brazilian regions
4. **Network Structure**: High connectivity (11.24% density) suggests rapid transmission potential

## Applications

- **Public Health Surveillance**: Prioritize monitoring in identified hub municipalities
- **Outbreak Prediction**: Use hub activity as early warning indicators
- **Intervention Planning**: Focus vector control on bridge municipalities
- **Resource Allocation**: Direct resources to high-impact transmission points

## References

- Discrete Morse Theory for network analysis
- SINAN (Sistema de Informação de Agravos de Notificação) data
- NetworkX for graph analysis
- Pandas for data manipulation

## Contributing

This analysis framework can be extended for:

- Other vector-borne diseases (Zika, Chikungunya)
- Different geographic scales (states, neighborhoods)
- Real-time surveillance integration
- Machine learning model integration

## License

This project is for research and public health applications. Please ensure compliance with data privacy regulations when working with health data.

---

**Note**: This analysis is based on 2023-2025 dengue data from SINAN. Results should be validated with epidemiological experts and historical outbreak patterns.
