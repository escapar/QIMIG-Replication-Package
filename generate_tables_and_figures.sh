#!/usr/bin/env bash

echo "================================================================="
echo "  QIMIG Replication Package: Automated Figure and Table Generator"
echo "================================================================="
echo ""

mkdir -p tables_and_figures

cd analysis || exit

echo "[*] 1/3: Generating Table 1 and Table 2 (Global & Per-Rule Performance) ..."
python -X utf8 generate_tables_1_and_2.py > ../tables_and_figures/output_table1_table2.txt
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to run generate_tables_1_and_2.py"
else
    echo "   -> Done! Output saved to: tables_and_figures/output_table1_table2.txt"
fi
echo ""

echo "[*] 2/3: Generating Statistical Significance Tests ..."
python -X utf8 test_statistical_significance.py > ../tables_and_figures/output_significance.txt
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to run test_statistical_significance.py"
else
    echo "   -> Done! Output saved to: tables_and_figures/output_significance.txt"
fi
echo ""

echo "[*] 3/3: Generating Parameter Sensitivity Plot (RQ3) ..."
python -X utf8 plot_sensitivity.py > ../tables_and_figures/output_sensitivity_text.txt
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to run plot_sensitivity.py"
else
    echo "   -> Done! Plot saved to: tables_and_figures/sensitivity_analysis_hybrid_pop.pdf"
    echo "   -> Done! LaTeX corpus saved to: tables_and_figures/output_sensitivity_text.txt"
fi
echo ""

cd ..
echo "================================================================="
echo "  Generation Complete!"
echo "  All tables and figures are located in the 'tables_and_figures' directory."
echo "================================================================="
