rm -rf logger.log
python pandapcd_si618_hw4_part2.py ./bills -o ./output
cat output/part* > pandapcd_si618_hw4_output_part2.txt
rm -rf output
