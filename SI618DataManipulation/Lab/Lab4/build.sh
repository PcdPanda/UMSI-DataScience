rm -rf si618_lab4_output
python pandapcd_si618_lab4.py ./bills -o si618_lab4_output
cat si618_lab4_output/part* > pandapcd_si618_lab4_output.tsv
rm -rf si618_lab4_output