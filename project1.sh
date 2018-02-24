#for ((i=30;i<100;i+=10));do
##	echo $i
    #python project1_Main.py $i
#done
fname="results.txt"
if [ -f fname ]; then
	rm $fname
fi
for ((i=30;i<100;i+=10));do
#	echo $i
	python project1_Main.py $i $fname
done