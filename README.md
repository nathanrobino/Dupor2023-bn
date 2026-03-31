# Dupor2023-bn Replication Package

**Paper**: *Welfare and Spending Effects of Consumption Stimulus Policies*  
**Authors**: Christopher D. Carroll, Edmund Crawley, William Du, Ivan Frankovic, Håkon Tretvoll  
**Keywords**: heterogeneous agents, fiscal policy, stimulus checks, iMPCs, HANK, consumption, welfare, QE replication
**Version**: Development Version (HAFiscal-Latest)

---

## File Organization (simplified)

```
/
├── README.md							 # This file
├── environment.yml						 # Conda environment specification
├── pyproject.toml						 # Python dependencies (uv format)
├── HAFiscal.tex						 # Main LaTeX document
├── HAFiscal.bib						 # Bibliography
├── HAFiscal-Abstract.txt				 # Abstract text
├── HAFiscal-Slides.tex					 # Presentation slides
├── reproduce.sh						 # Main reproduction script
├── reproduce.py						 # Python mirror (cross-platform)
├── reproduce_min.sh					 # Quick validation test
├── reproduce/							 # Additional reproduction scripts
│   ├── reproduce_computed.sh			 # Run all computations
│   ├── reproduce_computed_min.sh		 # Minimal computation test
│   ├── reproduce_documents.sh			 # Generate LaTeX documents
│   ├── reproduce_environment_comp_uv.sh # Set up Python environment (uv)
│   ├── reproduce_environment_texlive.sh # Set up LaTeX environment
│   └── [other reproduction scripts]
├── Code/								 # All computational code
│   ├── HA-Models/						 # Heterogeneous agent models
│   │   ├── FromPandemicCode/			 # Core model implementation
│   │   ├── Results/					 # Model output files
│   │   └── [model-specific directories]
│   └── Empirical/						 # Empirical data processing
│       ├── download_scf_data.sh		 # Download SCF data
│       ├── make_liquid_wealth.py		 # Construct liquid wealth measure
│       ├── adjust_scf_inflation.py		 # Inflation adjustments
│       └── compare_scf_datasets.py		 # Dataset comparisons
├── Figures/							 # Figure LaTeX files (*.tex, *.pdf)
├── Tables/								 # Table LaTeX files (*.tex, *.pdf)
├── Subfiles/							 # Paper section files
│   ├── Appendix-*.tex					 # Appendix sections
│   ├── Conclusion.tex					 # Conclusion section
│   └── [other section files]
├── Data/								 # Data files directory
├── dashboard/							 # Interactive dashboard (Jupyter/Streamlit)
├── binder/								 # Binder configuration for cloud execution
├── Equations/							 # Equation definitions
├── @local/								 # Local LaTeX packages and configuration
└── @resources/							 # LaTeX resources and utilities
```
---

**Last Updated**: March 31, 2026  
**README Version**: 1.1  
**Replication Package Version**: 1.0
