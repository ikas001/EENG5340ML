<<<<<<< HEAD
for ((i=30;i<100;i+=10));do
#	echo $i
    python project1_Main.py $i
=======
fname="results.txt"
if [ -f fname ] then
	rm $fname

for ((i=30;i<70;i+=10));do
#	echo $i
    python project1.py $i $fname
>>>>>>> Main/master
done
